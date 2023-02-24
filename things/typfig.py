

import numpy as np
from .core import try_this_wrap

import torch
from torch import Tensor


AnyTensor = np.ndarray | Tensor | list


@try_this_wrap(msg= ':x:to_numpy_safe')
def to_numpy_safe(tensor: Tensor | np.ndarray | list, dtype= np.float64, like: np.ndarray= None) -> np.ndarray:
	
	if isinstance(tensor, Tensor):
		tensor = tensor.detach().cpu().numpy()

	if like is not None:
		dtype = like.dtype

	if isinstance(tensor, (np.ndarray, np.generic)):
		tensor = list(tensor.tolist())
	
	return np.array(tensor).astype(dtype)


@try_this_wrap(msg= ':x:to_torch_safe')
def to_torch_safe(
	tensor: Tensor | np.ndarray | list, *, 
	dtype= torch.float32, device= 'cpu', requires_grad: bool= False, 
	like: Tensor= None) -> Tensor:
	
	if isinstance(tensor, (np.ndarray, np.generic)):
		tensor = list(tensor.tolist())

	if isinstance(tensor, list):
		tensor = np.array(tensor)

	if like is not None:
		device = like.device
		dtype = like.dtype
	
	return torch.tensor(tensor, dtype= dtype, device= device).requires_grad_(requires_grad)

from .core import flat_any

def get_el0(tensor: np.ndarray):
	if isinstance(tensor, AnyTensor):
		return tensor.item()
	elif isinstance(tensor, list):
		return flat_any(tensor)[0]
	return tensor

def convert_to(
	tensor: dict | np.ndarray | Tensor= None, *,
	to= 'numpy', device= 'cpu', dtype: str|type= 'float64', requires_grad: bool= False,
	like= None
):
	if not isinstance(tensor, (dict, list, AnyTensor)):
		return tensor

	if isinstance(tensor, dict):
		for k,v in tensor.items():
			tensor[k] = convert_to(v, to= to, device= device, dtype= dtype, requires_grad= requires_grad, like= like)
	else:
		dtype_all = dict(
			numpy= dict(
				float64	= np.float64,
				float32	= np.float32,
				int64	= np.int64,
				int32	= np.int32,
				uint8	= np.uint8,
			),
			torch= dict(
				float64	= torch.float64,
				float32	= torch.float32,
				int64	= torch.int64,
				int32	= torch.int32,
				bool	= torch.bool,
				uint8	= torch.uint8,
			)
		)

		if like is not None:
			if isinstance(like, Tensor):
				to = 'torch'
			elif isinstance(like, (np.ndarray, np.generic)):
				to = 'numpy'
		
		if to == 'np':
			to = 'numpy'
		elif to in ('tc', 'th', 'T'):
			to = 'torch'

		if isinstance(dtype, str):
			dtype = dtype_all[to][dtype]

		tensor = [] if tensor is None else tensor
		if np.isscalar(tensor):
			return tensor

		if len(tensor) == 0:
			if isinstance(tensor, dict):
				return {}
			elif isinstance(tensor, list):
				return []
		
		t = tensor
		if isinstance(t, (np.ndarray, np.generic, Tensor)):
			t = t.flatten().tolist()
		if isinstance(t, str):
			return tensor

		if to == 'numpy':
			return to_numpy_safe(tensor, dtype= dtype, like= like)
		elif to == 'torch':
			return to_torch_safe(tensor, dtype= dtype, device= device, requires_grad= requires_grad, like= like)
		else:
			raise ValueError(f'convert_to: invalid to: {to}')
	return tensor