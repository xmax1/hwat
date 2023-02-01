
from torch import nn
import torch 

def ke_fn_b(model: nn.Module, r: torch.Tensor, ke_method='jvp', elements=False):
	dtype, device = r.dtype, r.device
 
	n_b, n_e, n_dim = r.shape
	n_jvp = n_e * n_dim
	r_flat = r.reshape(n_b, n_jvp)
	eye = torch.eye(n_jvp, device=device, dtype=dtype, requires_grad=True)
	ones = torch.ones((n_b, ), device=device, dtype=dtype, requires_grad=True)

	def grad_fn(_r: torch.Tensor):
		print('r', _r.shape)
		_r.requires_grad_(True)
		[p.requires_grad_(False) for p in model.parameters()]
		lp = model(_r).sum()
		return torch.autograd.grad(lp, _r, create_graph=True)[0]

	def grad_grad_fn(_r: torch.Tensor):
		g = grad_fn(_r)
		ggs = [torch.autograd.grad(g[:, i], _r, grad_outputs=ones, retain_graph=True)[0] for i in range(n_jvp)]
		ggs = torch.stack(ggs, dim=-1)
		return torch.diagonal(ggs, dim1=1, dim2=2)

	g = grad_fn(r_flat)
	gg = grad_grad_fn(r_flat)
 
	if elements:
		return g, gg

	e_jvp = gg + g**2
	return -0.5 * e_jvp.sum(-1)

	return (gg + g**2)*-0.5
	
 
 
	from torch.autograd.functional import jacobian
	hess = torch.zeros((n_b,), device=device, dtype=dtype)
	print(grad_fn(r_flat)[None, :, 0].shape)
	for i in range(n_jvp):
		print(i)
		jac = jacobian(lambda _r: grad_fn(_r)[:, i], r_flat, vectorize=True)
		print(jac.shape)
		hess += jac[:, i]
	print(hess.shape)
	# model_hess = lambda r: model(r).sum()
	# hess = torch.autograd.functional.hessian(model_hess, r_flat, create_graph=False, strict=False, vectorize=True, outer_jacobian_strategy='reverse-mode')
	# print(hess.shape)
	return -0.5*(torch.diagonal(hess, dim1=1, dim2=2).sum(-1) + g**2)

	# eye_b = torch.eye(n_jvp, dtype=dtype, device=device).unsqueeze(0).repeat((n_b, 1, 1))
	# grad_fn = grad(lambda _r: model(_r).sum())
	# primal_g, fn = vjp(grad_fn, r_flat)
	# lap = torch.stack([fn(eye_b[..., i])[0] for i in range(n_jvp)], -1)
 
	
	# torch.autograd.functional.jacobian(func, inputs, create_graph=False, strict=False, vectorize=False, strategy='reverse-mode')
	# torch.autograd.functional.jacobian(func, inputs, create_graph=False, strict=False, vectorize=False, strategy='forward-mode')
	#  (primal[:, i]**2).squeeze() + (tangent[:, i]).squeeze()
	# primal, tangent = jax.jvp(grad_fn, (r,), (eye[..., i],))  
	# 	return val + (primal[:, i]**2).squeeze() + (tangent[:, i]).squeeze()
	# return (- 0.5 * jax.lax.fori_loop(0, n_jvp, _body_fun, jnp.zeros(n_b,))).squeeze()
	
	return (torch.diagonal(gg, dim1=1, dim2=2) + g**2).sum(-1)



# # Method 1
# # Use PyTorch autograd reverse mode + `for` loop.
# rev_jacobian = []
# # 1 forward pass.
# output = predict(weight, bias, primal)
# # M backward pass.
# for cotangent in cotangents:
#     # Compute vjp, where v = cotangent
#     (jacobian_row, ) = torch.autograd.grad(outputs=(output, ),
#                                            inputs=(primal, ),
#                                            grad_outputs=(cotangent, ),
#                                            retain_graph=True)
#     rev_jacobian.append(jacobian_row)
# # jacobian: [M, B, N]
# jacobian = torch.stack(rev_jacobian)
# # jacobian: [B, M, N]
# jacobian = jacobian.transpose(1, 0)

	# https://discuss.pytorch.org/t/error-by-recursively-calling-jacobian-in-a-for-loop/125924
"""
N in 
M out
dM / dN
N > M reverse
N < M forward 
* Do reverse mode followed by forward mode
* Reverse mode = vector jacobian product
* Forward mode = jacobian vector product

# HWAT 
Forward: n_e * 3 -> 1
Grad: n_e * 3 -> n_e * 3


Q: Why not just compute Hessian? 
Computing the diagonal of the Hessian is cheaper in some cases
https://timvieira.github.io/blog/post/2014/02/10/gradient-vector-product/


Potential Crap:
- # https://j-towns.github.io/2017/06/12/A-new-trick.html 
"""

# def fd2(r_flat, step=0.00001):
#     model_fd2 = lambda _r: model_fn(params, _r)
#     model_fd2 = vmap(vmap(vmap(model_fd2)))
#     r_flat = r_flat.unsqueeze(1)
#     step_eye = (torch.eye(n_e*3, dtype=r_flat.dtype)*step).unsqueeze(0)
#     r_flat0 = torch.stack([r_flat-step_eye, torch.tile(r_flat, (1, n_e*3, 1)), r_flat+step_eye], dim=0)
#     print(r_flat0.shape)
#     log_psi = model_fd2(r_flat0)
#     factor = torch.tensor([+1, -2, +1]).unsqueeze(-1).unsqueeze(-1)
#     print(factor.shape, log_psi.shape)
#     return (factor * log_psi).sum(0) / step**2


def compute_fd_g(fn, x, dx):
    device = x.device
    n_b, n_f = x.shape
    step = torch.eye(n_f, dtype=r_flat.dtype, device=device).unsqueeze(0)*dx
    # x_line = torch.stack([-(x[..., None] -step), x[..., None, :]+step])
    # grad = x_line.sum(0) / (2*step)
    out_neg = fn((x.unsqueeze(-1)-step).swapaxes(-1,-2).reshape(-1, n_f))
    out_pos = fn((x.unsqueeze(-1)+step).swapaxes(-1,-2).reshape(-1, n_f))
    grad = (out_pos-out_neg) / (2*dx)
    return grad.reshape(n_b, n_f)
    return torch.stack([x-step, x+step])

def check(a, b, name=None):
	if not torch.allclose(a, b):
		print('\n', name)
		print('a / b: ')
		print(a.shape, b.shape)
		print(a, '\n', b)
  
if __name__ == "__main__":
	
	import torch 
	import numpy as np
	torch.manual_seed(1)
	torch.set_default_tensor_type(torch.DoubleTensor)   # ❗ Ensure works when default not set AND can go float32 or 64

	from hwat_func import init_r, get_center_points

	### pyfig ###
	arg = dict(
		charge = 0,
		spin  = 0,
		a = np.array([[0.0, 0.0, 0.0],]),
		a_z  = np.array([4.,]),
		n_b = 2, 
		n_sv = 32, 
		n_pv = 16, 
		n_corr = 40, 
		n_step = 10000, 
		log_metric_step = 5, 
		exp_name = 'demo',
		# sweep = {},
	)

	from pyfig import Pyfig
	from hwat_b import Ansatz_fb as F_b
	from hwat_func import Ansatz_fb as F
	from hwat_func import compute_ke_b
	from functorch import vmap, make_functional, grad   

	c = Pyfig(wb_mode='disabled', arg=arg, submit=False, run_sweep=False, run_name='debug_ke.py')
	
	_dummy = torch.randn((1,))
	dtype = _dummy.dtype
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	c._convert(device=device, dtype=dtype)
 
	center_points = get_center_points(c.data.n_e, c.app.a)
	n_b, n_e = c.data.n_b, c.data.n_e
	r = init_r(n_b, n_e, center_points, std=0.1)
	r_flat = r.reshape(n_b, -1)
	deltar = torch.tensor([0.02], device=device, dtype=dtype)
	print('r', r.shape, r_flat.shape)

	model = c.partial(F_b, with_sign=True).to(device).to(dtype)
	lp, sn = model(r)

	model_fn = c.partial(F, with_sign=True)

	# with torch.no_grad():
	# 	for (name, p), p_func in zip(model.named_parameters(), model_fn.parameters()):
	# 		p_func = p.detach().clone()

	params = [p.detach().clone() for p in model.parameters()]
 
	model_fn, _ = make_functional(model_fn)
	model_v = vmap(model_fn, in_dims=(None, 0))
	lp_fn, sn_fn = model_v(params, r) 
 
	[check(p, p_func, name) for (name, p), p_func in zip(model.named_parameters(), params)]

	check(lp, lp_fn, 'lp')
	check(sn, sn_fn, 'sign')
	# print(lp - lp_fn, sn - sn_fn)

	model = c.partial(F_b).to(device).to(dtype)
 
	model_fn = c.partial(F)
	model_fn, _ = make_functional(model_fn)
	params = [p.detach().clone() for p in model.parameters()]
	model_fn_partial = lambda _r: model_fn(params, _r)
	model_v = vmap(model_fn, in_dims=(None, 0))
	model_ke = lambda _r: model_v(params, _r).sum()
 

	ke = ke_fn_b(model, r)
	g, gg = ke_fn_b(model, r, elements=True)

	ke_func = compute_ke_b(model_ke, r)
	g_func_vjp, gg_func_vjp = compute_ke_b(model_ke, r, elements=True, ke_method='vjp')
	g_func_jvp, gg_func_jvp = compute_ke_b(model_ke, r, elements=True, ke_method='jvp')
 
	# G, GG, difftorch, num
	import difftorch
	b = 1
	fd_step = 0.000001
	gg_diff = difftorch.laplacian(model_fn_partial, r_flat[b])
	g_fd = compute_fd_g(model, r_flat, fd_step)
 
	if not torch.allclose(gg_func_vjp[b].sum(), gg_diff):
		print('diff / func: \n', gg_func_vjp[b].sum(), '\n', gg_diff)
	if not torch.allclose(gg_func_jvp[b].sum(), gg_diff):
		print('diff / func: \n', gg_func_jvp[b].sum(), '\n', gg_diff)

	check(g, g_func_vjp, 'g')
	check(g_fd, g_func_vjp, 'g')
	check(gg, gg_func_vjp)
	# print(g - g_func, gg - gg_func)
	# print(ke - ke_func)