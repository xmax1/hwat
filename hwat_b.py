from functools import reduce
import torch 
import torch.nn as nn

def logabssumdet(xs):

    dets = [x.reshape(x.shape[0], -1) for x in xs if x.shape[-1] == 1]				# in case n_u or n_d=1, no need to compute determinant
    dets = reduce(lambda a,b: a*b, dets) if len(dets)>0 else 1.					# take product of these cases (equiv to dets = torch.prod(torch.cat(dets, dim=1), dim=1) if len(dets)>0 else 1.)
    maxlogdet = 0.																# initialised for sumlogexp trick (for stability)
    det = dets																	# if both cases satisfy n_u or n_d=1, this is the determinant
    
    slogdets = [torch.linalg.slogdet(x) for x in xs if x.shape[-1]>1] 			# otherwise take slogdet
    if len(slogdets)>0: 
        sign_in, logdet = reduce(lambda a,b: (a[0]*b[0], a[1]+b[1]), slogdets)  # take product of n_u or n_d!=1 cases
        maxlogdet = torch.max(logdet)											# adjusted for new inputs
        det = sign_in * dets * torch.exp(logdet-maxlogdet)						# product of all these things is determinant
    
    psi_ish = torch.sum(det[:, None], dim=1)									# sum over determinants (n_batch
    sgn_psi = torch.sign(psi_ish)
    log_psi = torch.log(torch.abs(psi_ish)) + maxlogdet
    return log_psi, sgn_psi


class Ansatz_fb(nn.Module):
	def __init__(self, n_e, n_u, n_d, n_det, n_fb, n_pv, n_sv, a: torch.Tensor, with_sign=False):
		super(Ansatz_fb, self).__init__()
		self.n_e = n_e                  # number of electrons
		self.n_u = n_u                  # number of up electrons
		self.n_d = n_d                  # number of down electrons
		self.n_det = n_det              # number of determinants
		self.n_fb = n_fb                # number of feedforward blocks
		self.n_pv = n_pv                # latent dimension for 2-electron
		self.n_sv = n_sv                # latent dimension for 1-electron
		self.a = a                      # nuclei positions
		self.with_sign = with_sign      # return sign of wavefunction

		self.n1 = [4*self.a.shape[0]] + [self.n_sv]*self.n_fb
		self.n2 = [4] + [self.n_pv]*(self.n_fb - 1)
		assert (len(self.n1) == self.n_fb+1) and (len(self.n2) == self.n_fb)
		self.Vs = nn.ModuleList([nn.Linear(3*self.n1[i]+2*self.n2[i], self.n1[i+1]) for i in range(self.n_fb)])
		self.Ws = nn.ModuleList([nn.Linear(self.n2[i], self.n2[i+1]) for i in range(self.n_fb)])

		self.V_half_u = nn.Linear(self.n_sv, self.n_sv // 2)
		self.V_half_d = nn.Linear(self.n_sv, self.n_sv // 2)

		self.wu = nn.Linear(self.n_sv // 2, self.n_u)
		self.wd = nn.Linear(self.n_sv // 2, self.n_d)

	def forward(self, r: torch.Tensor):
		"""
			Takes in a tensor of shape (n_batch, n_e, 3) and returns the log of the wavefunction and the sign of the wavefunction.
		"""
		if len(r.shape)==2:
			r = r[None, ...]
		n_batch = r.shape[0]

		eye = torch.eye(self.n_e, device=device, dtype=dtype).unsqueeze(-1)

		ra = r[:, :, None, :] - self.a[:, :] # (n_batch, n_e, n_a, 3)
		ra_len = torch.norm(ra, dim=-1, keepdim=True) # (n_batch, n_e, n_a, 1)

		rr = r[:, None, :, :] - r[:, :, None, :] # (n_batch, n_e, n_e, 1)
		rr_len = torch.norm(rr+eye, dim=-1, keepdim=True) * (torch.ones((self.n_e, self.n_e, 1))-eye) # (n_batch, n_e, n_e, 1) 

		s_v = torch.cat([ra, ra_len], dim=-1).reshape(n_batch, self.n_e, -1) # (n_batch, n_e, n_a*4)
		p_v = torch.cat([rr, rr_len], dim=-1) # (n_batch, n_e, n_e, 4)

		for l, (V, W) in enumerate(zip(self.Vs, self.Ws)):
			sfb_v = [torch.tile(_v.mean(dim=1)[:, None, :], (self.n_e, 1)) for _v in torch.split(s_v, self.n_u, dim=1)] # two element list of (n_batch, n_e, n_sv) tensor
			pfb_v = [_v.mean(dim=1) for _v in torch.split(p_v, self.n_u, dim=1)] # two element list of (n_batch, n_e, n_pv) tensor

			s_v = torch.cat(sfb_v+pfb_v+[s_v,], dim=-1) # (n_batch, n_e, 3n_sv+2n_pv)
			s_v = torch.tanh(V(s_v)) + (s_v if (s_v.shape[-1]==self.n_sv) else 0.) # (n_batch, n_e, n_sv)

			if not (l == (self.n_fb-1)):
				p_v = torch.tanh(W(p_v)) + (p_v if (p_v.shape[-1]==self.n_pv) else 0.) # (n_batch, n_e, n_e, n_pv)


		s_u, s_d = torch.split(s_v, self.n_u, dim=1) # (n_batch, n_u, n_sv), (n_batch, n_d, n_sv)

		s_u = torch.tanh(self.V_half_u(s_u)) # (n_batch, n_u, n_sv//2)
		s_d = torch.tanh(self.V_half_d(s_d)) # (n_batch, n_d, n_sv//2)

		s_wu = self.wu(s_u) # (n_batch, n_u, n_u)
		s_wd = self.wd(s_d) # (n_batch, n_d, n_d)

		assert s_wd.shape == (n_batch, self.n_d, self.n_d)

		ra_u, ra_d = torch.split(ra, self.n_u, dim=1) # (n_batch, n_u, n_a, 3), (n_batch, n_d, n_a, 3)

		exp_u = torch.norm(ra_u, dim=-1, keepdim=True) # (n_batch, n_u, n_a, 1)
		exp_d = torch.norm(ra_d, dim=-1, keepdim=True) # (n_batch, n_d, n_a, 1)

		assert exp_d.shape == (n_batch, self.n_d, self.a.shape[0], 1)

		orb_u = (s_wu * (torch.exp(-exp_u).sum(axis=2))) # (n_batch, 1, n_u, n_u)
		orb_d = (s_wd * (torch.exp(-exp_d).sum(axis=2))) # (n_batch, 1, n_d, n_d)

		assert orb_u.shape == (n_batch, self.n_u, self.n_u) # extend with n_det axis in dim=1

		log_psi, sgn = logabssumdet([orb_u, orb_d])

		if self.with_sign:
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
	pe_rr += torch.tril(1./rr_len, diagonal=-1).sum((-1,-2))

	if not a is None:
		a, a_z = a[None, :, :], a_z[None, None, :]
		ra = torch.unsqueeze(r, -2) - torch.unsqueeze(a, -3)
		ra_len = torch.linalg.norm(ra, axis=-1)
		pe_ra += (a_z/ra_len).sum((-1,-2))

		if len(a_z) > 1:
			aa = torch.unsqueeze(a, -2) - torch.unsqueeze(a, -3)
			aa_len = torch.linalg.norm(aa, axis=-1)
			pe_aa += torch.tril(1./aa_len, diagonal=-1).sum((-1,-2))

	return (pe_rr - pe_ra + pe_aa).squeeze()  

def compute_ke_b(model_fnv: nn.Module, r: torch.Tensor):
	dtype, device = r.dtype, r.device
	# MODEL IS FUNCTIONAL THAT IS VMAPPED
	n_b, n_e, n_dim = r.shape
	n_jvp = n_e * n_dim
	r_flat = r.reshape(n_b, n_jvp)
	# print(r.shape, r.dtype)
	eye_b = torch.eye(n_jvp, dtype=dtype, device=device).unsqueeze(0).repeat((n_b, 1, 1))
	grad_fn = grad(lambda _r: model_fnv(_r).sum())
	primal_g, fn = vjp(grad_fn, r_flat)
	lap = torch.stack([fn(eye_b[..., i])[0] for i in range(n_jvp)], -1)
 
	#  (primal[:, i]**2).squeeze() + (tangent[:, i]).squeeze()
	# primal, tangent = jax.jvp(grad_fn, (r,), (eye[..., i],))  
	# 	return val + (primal[:, i]**2).squeeze() + (tangent[:, i]).squeeze()
	# return (- 0.5 * jax.lax.fori_loop(0, n_jvp, _body_fun, jnp.zeros(n_b,))).squeeze()
	
	return (torch.diagonal(lap, dim1=1, dim2=2) + primal_g**2).sum(-1)

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
	r = r - points[None, ...]
	r = r/l
	r = torch.fmod(r, 1.)
	r = r*l
	r = r + points[None, ...]
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

def sample_b(model, r_0, deltar_0, n_corr=10):
	""" metropolis hastings sampling with automated step size adjustment """
	device, dtype = r_0.device, r_0.dtype
	deltar_1 = torch.clip(deltar_0 + 0.01*torch.randn([1,], device=device, dtype=dtype), min=0.005, max=0.5)

	acc = []
	for deltar in [deltar_0, deltar_1]:
		
		for _ in torch.arange(n_corr):

			p_0 = (torch.exp(model(r_0))**2)  			# ❗can make more efficient with where modelment at end
			
			# print(deltar.shape, r_0.shape)
			r_1 = r_0 + torch.randn_like(r_0, device=device, dtype=dtype)*deltar
			
			p_1 = torch.exp(model(r_1))**2
			# p_1 = torch.where(torch.isnan(p_1), 0., p_1)    # :❗ needed when there was a bug in pe, needed now?!

			p_mask = (p_1/p_0) > torch.rand_like(p_1, device=device, dtype=dtype)		# metropolis hastings
			
			r_0 = torch.where(p_mask[..., None, None], r_1, r_0)
	
		acc += [p_mask.type_as(r_0).mean()]
	
	mask = ((0.5-acc[0])**2 - (0.5-acc[1])**2) < 0.
	deltar = mask*deltar_0 + ~mask*deltar_1
	
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
