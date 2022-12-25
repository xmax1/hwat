from functools import reduce
import torch 
import torch.nn as nn

from functorch import vmap, grad, vjp 



class Ansatz_fb(nn.Module):
	def __init__(ii, n_e, n_u, n_d, n_det, n_fb, n_pv, n_sv, a: torch.Tensor, with_sign=False):
		super(Ansatz_fb, ii).__init__()
		ii.n_e = n_e                  # number of electrons
		ii.n_u = n_u                  # number of up electrons
		ii.n_d = n_d                  # number of down electrons
		ii.n_det = n_det              # number of determinants
		ii.n_fb = n_fb                # number of feedforward blocks
		ii.n_pv = n_pv                # latent dimension for 2-electron
		ii.n_sv = n_sv                # latent dimension for 1-electron
		ii.a = a       
		ii.n_a = len(a)               # nuclei positions
		ii.with_sign = with_sign      # return sign of wavefunction

		ii.n1 = [4*ii.n_a,] + [ii.n_sv,]*(ii.n_fb-1)
		ii.n2 = [4,] + [ii.n_pv,]*(ii.n_fb-1)
  
		ii.Vs = nn.ModuleList([nn.Linear(3*ii.n1[i]+2*ii.n2[i], ii.n1[i+1]) for i in range(ii.n_fb-1)])
		ii.Ws = nn.ModuleList([nn.Linear(ii.n2[i], ii.n2[i+1]) for i in range(ii.n_fb-1)])

		ii.after = nn.Linear(3*ii.n1[-1]+2*ii.n2[-1], ii.n1[-1])
  
		ii.V_half_u = nn.Linear(ii.n_sv, ii.n_sv // 2)
		ii.V_half_d = nn.Linear(ii.n_sv, ii.n_sv // 2)

		ii.wu = nn.Linear(ii.n_sv // 2, ii.n_u)
		ii.wd = nn.Linear(ii.n_sv // 2, ii.n_d)

		# TODO: Multideterminant. If n_det > 1 we should map to n_det*n_u (and n_det*n_d) instead,
		#  and then split these outputs in chunks of n_u (n_d)
		# TODO: implement layers for sigma and pi

	def forward(ii, r: torch.Tensor):
		
		dtype, device = r.dtype, r.device
  
		if len(r.shape) == 1:
			r = r.reshape(ii.n_e, 3) # (n_e, 3)

		# print(r.shape, ii.a.shape)
		eye = torch.eye(ii.n_e, device=device, dtype=dtype).unsqueeze(-1)

		ra = r[:, None, :] - ii.a[None, :, :] # (n_e, n_a, 3)
		ra_len = torch.linalg.norm(ra, dim=-1, keepdim=True) # (n_e, n_a, 1)

		rr = r[None, :, :] - r[:, None, :] # (n_e, n_e, 1)
		rr_len = torch.linalg.norm(rr + eye, dim=-1, keepdim=True) * (torch.ones((ii.n_e, ii.n_e, 1), device=device, dtype=dtype)-eye) # (n_e, n_e, 1) 
		# TODO: Just remove '+eye' from above, it's unnecessary

		s_v = torch.cat([ra, ra_len], dim=-1).reshape(ii.n_e, -1) # (n_e, n_a*4)
		p_v = torch.cat([rr, rr_len], dim=-1) # (n_e, n_e, 4)

		# print(s_v.mean(), p_v.mean())
		def fb_block(s_v, p_v):
			sfb_v = [torch.tile(_v.mean(dim=0)[None, :], (ii.n_e, 1)) for _v in torch.split(s_v, [ii.n_u,ii.n_d], dim=0)]
			pfb_v = [_v.mean(dim=0) for _v in torch.split(p_v, [ii.n_u,ii.n_d], dim=0)]
			s_v = torch.cat( sfb_v + pfb_v + [s_v,], dim=-1) # s_v = torch.cat((s_v, sfb_v[0], sfb_v[1], pfb_v[0], pfb_v[0]), dim=-1)
			return s_v
		
		for l, (V, W) in enumerate(zip(ii.Vs, ii.Ws)):
			s_v = fb_block(s_v, p_v)
   
			# print(l, s_v.mean())
			# print(s_v.dtype, p_v.dtype, s_v.shape, p_v.shape)
			s_v = torch.tanh(V(s_v)) + (s_v if (s_v.shape[-1]==ii.n_sv) else torch.zeros(ii.n_sv).type_as(s_v))
			p_v = torch.tanh(W(p_v)) + (p_v if (p_v.shape[-1]==ii.n_pv) else torch.zeros(ii.n_pv).type_as(p_v))
   
		# print(s_v.shape)
		s_v = fb_block(s_v, p_v)
		s_v = torch.tanh(ii.after(s_v)) + (s_v if (s_v.shape[-1]==ii.n_sv) else 0.)
		# print('2', s_v.mean())
		s_u, s_d = torch.split(s_v, [ii.n_u,ii.n_d], dim=0)

		s_u = ii.V_half_u(s_u) # spin dependent size reduction
		s_d = ii.V_half_d(s_d)

		s_wu = ii.wu(s_u) # map to phi orbitals
		s_wd = ii.wd(s_d)
		# print('s_wu', s_wu.mean())
		assert s_wd.shape == (ii.n_d, ii.n_d)

		ra_u, ra_d = torch.split(ra, [ii.n_u, ii.n_d], dim=0)

		# TODO: implement sigma = nn.Linear() before this
		exp_u = torch.linalg.norm(ra_u, dim=-1, keepdim=True)
		exp_d = torch.linalg.norm(ra_d, dim=-1, keepdim=True)
		# print('exp', exp_u.mean(), exp_d.mean())
		assert exp_d.shape == (ii.n_d, ii.a.shape[0], 1)

		# TODO: implement pi = nn.Linear() before this
		orb_u = (s_wu * (torch.exp(-exp_u).sum(axis=1)))[None, :, :]
		orb_d = (s_wd * (torch.exp(-exp_d).sum(axis=1)))[None, :, :]

		# print('exp', orb_u.mean(), orb_d.mean())
		assert orb_u.shape == (1, ii.n_u, ii.n_u)

		log_psi, sgn = logabssumdet([orb_u, orb_d])

		if ii.with_sign:
			return log_psi, sgn
		else:
			return log_psi.squeeze()
		
def logabssumdet(xs):
		
	dets = [x.reshape(-1) for x in xs if x.shape[-1] == 1]						# in case n_u or n_d=1, no need to compute determinant
	dets = reduce(lambda a,b: a*b, dets) if len(dets)>0 else 1.					# take product of these cases
	maxlogdet = 0.																# initialised for sumlogexp trick (for stability)
	det = dets																	# if both cases satisfy n_u or n_d=1, this is the determinant
	
	slogdets = [torch.linalg.slogdet(x) for x in xs if x.shape[-1]>1] 			# otherwise take slogdet
	if len(slogdets)>0: 
		sign_in, logdet = reduce(lambda a,b: (a[0]*b[0], a[1]+b[1]), slogdets)  # take product of n_u or n_d!=1 cases
		maxlogdet = torch.max(logdet)											# adjusted for new inputs
		det = sign_in * dets * torch.exp(logdet-maxlogdet)						# product of all these things is determinant
	
	psi_ish = torch.sum(det)
	sgn_psi = torch.sign(psi_ish)
	log_psi = torch.log(torch.abs(psi_ish)) + maxlogdet
	return log_psi, sgn_psi

compute_vv = lambda v_i, v_j: torch.unsqueeze(v_i, axis=-2)-torch.unsqueeze(v_j, axis=-3)

def compute_emb(r, terms, a=None):  
	dtype, device = r.dtype, r.device
	n_e, _ = r.shape
	eye = torch.eye(n_e)[..., None]

	z = []  
	if 'r' in terms:  
		z += [r]  
	if 'r_len' in terms:  
		z += [torch.linalg.norm(r, axis=-1, keepdims=True)]  
	if 'ra' in terms:  
		z += [(r[:, None, :] - a[None, ...]).reshape(n_e, -1)]  
	if 'ra_len' in terms:  
		z += [torch.linalg.norm(r[:, None, :] - a[None, ...], axis=-1)]
	if 'rr' in terms:
		z += [compute_vv(r, r)]
	if 'rr_len' in terms:  # 2nd order derivative of norm is undefined, so use eye
		z += [torch.linalg.norm(compute_vv(r, r)+eye, axis=-1, keepdims=True) * (torch.ones((n_e,n_e,1), device=device, dtype=dtype)-eye)]
	return torch.concatenate(z, axis=-1)

### energy ###

def compute_pe_b(r, a=None, a_z=None):
	dtype, device = r.dtype, r.device
 
	pe_rr = torch.zeros(r.shape[0], dtype=dtype, device=device)
	pe_ra = torch.zeros(r.shape[0], dtype=dtype, device=device)
	pe_aa = torch.zeros(r.shape[0], dtype=dtype, device=device)

	rr = torch.unsqueeze(r, -2) - torch.unsqueeze(r, -3)
	rr_len = torch.linalg.norm(rr, axis=-1)
	pe_rr += torch.tril(1./rr_len, diagonal=-1).sum((1,2))

	if not a is None:
		a, a_z = a[None, :, :], a_z[None, None, :]
		ra = torch.unsqueeze(r, -2) - torch.unsqueeze(a, -3)
		ra_len = torch.linalg.norm(ra, axis=-1)
		pe_ra += (a_z/ra_len).sum((1,2))

		if len(a_z) > 1:
			# print('here')
			# aa = torch.unsqueeze(a, -2) - torch.unsqueeze(a, -3)
			# aa_len = torch.linalg.norm(aa, axis=-1)
			# pe_aa += torch.tril((a_z*a_z)/aa_len, diagonal=-1).sum((-1,-2))
			raise NotImplementedError

	return (pe_rr - pe_ra + pe_aa).squeeze()  

from functorch import jvp 
from torch import jit

def compute_ke_b(model_rv, r: torch.Tensor):
	dtype, device = r.dtype, r.device
	# MODEL IS FUNCTIONAL THAT IS VMAPPED
	n_b, n_e, n_dim = r.shape
	n_jvp = n_e * n_dim
	r_flat = r.reshape(n_b, n_jvp)
	eyes = torch.eye(n_jvp, dtype=dtype, device=device)[None].repeat((n_b, 1, 1))

	# vjp method
	# grad_fn = grad(lambda _r: model_rv(_r).sum())
	# g, fn = vjp(grad_fn, r_flat)
	# gg2 = torch.stack([fn(eye_b[..., i])[0][:, i] for i in range(n_jvp)], dim=0).sum(0)
	# lap = gg2 + (g**2).sum(-1)
 
	# jvp 
	grad_fn = grad(model_rv)
	jvp_all = [jvp(grad_fn, (r_flat,), (eyes[:, i],)) for i in range(n_jvp)]  # grad out, jvp
	e_jvp = torch.stack([a[:, i]**2 + b[:, i] for i, (a,b) in enumerate(jvp_all)]).sum(0)
 
	#  (primal[:, i]**2).squeeze() + (tangent[:, i]).squeeze()
	# primal, tangent = jax.jvp(grad_fn, (r,), (eye[..., i],))  
	# 	return val + (primal[:, i]**2).squeeze() + (tangent[:, i]).squeeze()
	# return (- 0.5 * jax.lax.fori_loop(0, n_jvp, _body_fun, jnp.zeros(n_b,))).squeeze()

	return -0.5*e_jvp

# def compute_ke_b(model, r):
	
# 	grads = torch.autograd.grad(lambda r: model(r).sum(), r, create_graph=True)
	
# 	n_b, n_e, n_dim = r.shape
# 	n_jvp = n_e * n_dim
# 	r = r.reshape(n_b, n_jvp)
# 	eye = torch.eye(n_jvp, dtype=r.dtype)[None, ...].repeat(n_b, axis=0)
	
# 	def _body_fun(i, val):
# 		primal, tangent = jax.jvp(grad_fn, (r,), (eye[..., i],))  
# 		return val + (primal[:, i]**2).squeeze() + (tangent[:, i]).squeeze()
	
# 	return (- 0.5 * jax.lax.fori_loop(0, n_jvp, _body_fun, torch.zeros(n_b,))).squeeze()

### sampling ###
def keep_around_points(r, points, l=1.):
	""" points = center of box each particle kept inside. """
	""" l = length side of box """
	r = r - points[None]
	r = r/l
	r = torch.fmod(r, 1.)
	r = r*l
	r = r + points[None]
	return r

def get_center_points(n_e, center: torch.Tensor, _r_cen=None):
	""" from a set of centers, selects in order where each electron will start """
	""" loop concatenate pattern """
	for r_i in range(n_e):
		r_i = center[[r_i % len(center)]]
		_r_cen = r_i if _r_cen is None else torch.concatenate([_r_cen, r_i])
	return _r_cen

def init_r(n_device, n_b, n_e, center_points: torch.Tensor, std=0.1):
	dtype, device = center_points.dtype, center_points.device
	""" init r on different gpus with different rngs """
	""" loop concatenate pattern """
	sub_r = [center_points + torch.randn((n_b,n_e,3), device=device, dtype=dtype)*std for i in range(n_device)]
	return torch.stack(sub_r) if len(sub_r)>1 else sub_r[0][None, ...]

def sample_b(model, params, r_0, deltar_0, n_corr=10):
	""" metropolis hastings sampling with automated step size adjustment """
	device, dtype = r_0.device, r_0.dtype
	
	deltar_1 = torch.clip(deltar_0 + 0.01*torch.randn([1,], device=device, dtype=dtype), min=0.005, max=0.5)

	acc = []
	for deltar in [deltar_0, deltar_1]:
		
		for _ in torch.arange(n_corr):

			p_0 = torch.exp(model(params, r_0))**2				# ❗can make more efficient with where modelment at end
			
			# print(deltar.shape, r_0.shape)
			r_1 = r_0 + torch.randn_like(r_0, device=device, dtype=dtype)*deltar
			
			p_1 = torch.exp(model(params, r_1))**2
			# p_1 = torch.where(torch.isnan(p_1), 0., p_1)    # :❗ needed when there was a bug in pe, needed now?!

			p_mask = (p_1/p_0) > torch.rand_like(p_1, device=device, dtype=dtype)		# metropolis hastings
			
			r_0 = torch.where(p_mask[:, None, None], r_1, r_0)
	
		acc += [p_mask.type(dtype).mean()]
	
	mask = ((0.5-acc[0])**2 - (0.5-acc[1])**2) < 0.
	deltar = mask.to(dtype)*deltar_0 + (~mask).to(dtype)*deltar_1
	
	return r_0, (acc[0]+acc[1])/2., deltar

### Test Suite ###

# def check_antisym(c, r):
# 	n_u, n_d, = c.data.n_u, c.data.n_d
# 	r = r[:, :4]
	
# 	@partial(jax.vmap, in_axes=(0, None, None))
# 	def swap_rows(r, i_0, i_1):
# 		return r.at[[i_0,i_1], :].set(r[[i_1,i_0], :])

# 	@partial(jax.pmap, axis_name='dev', in_axes=(0,0))
# 	def _create_train_model(r):
# 		model = c.partial(FermiNet, with_sign=True)  
# 		params = model.init(r)['params']
# 		return TrainState.create(apply_fn=model.apply, params=params, tx=c.opt.tx)
	
# 	model = _create_train_model(r)

# 	@partial(jax.pmap, in_axes=(0, 0))
# 	def _check_antisym(model, r):
# 		log_psi_0, sgn_0 = model.apply_fn(model.params, r)
# 		r_swap_u = swap_rows(r, 0, 1)
# 		log_psi_u, sgn_u = model.apply_fn(model.params, r_swap_u)
# 		log_psi_d = torch.zeros_like(log_psi_0)
# 		sgn_d = torch.zeros_like(sgn_0)
# 		if not n_d == 0:
# 			r_swap_d = swap_rows(r, n_u, n_u+1)
# 			log_psi_d, sgn_d = model.apply_fn(model.params, r_swap_d)
# 		return (log_psi_0, log_psi_u, log_psi_d), (sgn_0, sgn_u, sgn_d), (r, r_swap_u, r_swap_d)

# 	res = _check_antisym(model, r)

# 	(log_psi, log_psi_u, log_psi_d), (sgn, sgn_u, sgn_d), (r, r_swap_u, r_swap_d) = res
# 	for ei, ej, ek in zip(r[0,0], r_swap_u[0,0], r_swap_d[0,0]):
# 		print(ei, ej, ek)  # Swap Correct
# 	for lpi, lpj, lpk in zip(log_psi[0], log_psi_u[0], log_psi_d[0]):
# 		print(lpi, lpj, lpk)  # Swap Correct
# 	for lpi, lpj, lpk in zip(sgn[0], sgn_u[0], sgn_d[0]):
# 		print(lpi, lpj, lpk)  # Swap Correct


# def logabssumdet(xs):
	
# 	dets = [x.reshape(-1) for x in xs if x.shape[-1] == 1]						# in case n_u or n_d=1, no need to compute determinant
# 	dets = reduce(lambda a,b: a*b, dets) if len(dets)>0 else 1.					# take product of these cases
# 	maxlogdet = 0.																# initialised for sumlogexp trick (for stability)
# 	det = dets																	# if both cases satisfy n_u or n_d=1, this is the determinant
	
# 	slogdets = [torch.linalg.slogdet(x) for x in xs if x.shape[-1]>1] 			# otherwise take slogdet
# 	if len(slogdets)>0: 
# 		sign_in, logdet = reduce(lambda a,b: (a[0]*b[0], a[1]+b[1]), slogdets)  # take product of n_u or n_d!=1 cases
# 		maxlogdet = torch.max(logdet)												# adjusted for new inputs
# 		det = sign_in * dets * torch.exp(logdet-maxlogdet)						# product of all these things is determinant
	
# 	psi_ish = torch.sum(det)
# 	sgn_psi = torch.sign(psi_ish)
# 	log_psi = torch.log(torch.abs(psi_ish)) + maxlogdet
# 	return log_psi, sgn_psi
