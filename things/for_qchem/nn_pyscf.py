from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
from torch import nn

import torch


import torch
from torch import nn


from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax


from things.for_jax.utils import pmean
from nn import MLP, ParamTree
from systems.scf import Scf


def eval_orbitals(
	scf_approx: List[Scf], 
	electrons: jnp.ndarray, 
	spins: Tuple[int, int], 
	signs: Tuple[np.ndarray, np.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the molecular orbitals of Hartree Fock calculations.
    Args:
        scf_approx (List[Scf]): Hartree Fock calculations, length H
        electrons ([type]): (H, B, N, 3)
        spins ([type]): (spin_up, spin_down)
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: [(H, B, spin_up, spin_up), (H, B, spin_down, spin_down)] molecular orbitals
    """
	
    if isinstance(scf_approx, (list, tuple)):
        n_scf = len(scf_approx)
    else:
        n_scf = 1
        scf_approx = [scf_approx]

    leading_dims = electrons.shape[:-1]
    electrons = electrons.reshape([n_scf, -1, 3])  # (batch*nelec, 3)
    # (batch*nelec, nbasis), (batch*nelec, nbasis)

    mos = [scf.eval_molecular_orbitals(e) for scf, e in zip(scf_approx, electrons)]


    mos = (np.stack([mo[0] for mo in mos], axis=0),
           np.stack([mo[1] for mo in mos], axis=0))
    # Reshape into (batch, nelec, nbasis) for each spin channel
    mos = [mo.reshape(leading_dims + (sum(spins), -1)) for mo in mos]

    alpha_spin = mos[0][..., :spins[0], :spins[0]]
    beta_spin = mos[1][..., spins[0]:, :spins[1]]

    # Adjust signs
    if signs is not None:
        alpha_signs, beta_signs = signs
        alpha_spin[..., 0] *= alpha_signs[..., None, None]
        beta_spin[..., 0] *= beta_signs[..., None, None]
    return alpha_spin, beta_spin


def determine_hf_signs(scfs: List[Scf], electrons: jnp.ndarray, spins: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    n_scfs = len(scfs)
    # Compute orbitals
    # We use some samples from each configuration for this
    samples = np.asarray(electrons).reshape(-1, electrons.shape[-1])
    up_down_orbitals = eval_orbitals(scfs, np.broadcast_to(
        samples, (*electrons.shape[:2], *samples.shape)), spins)
    # Compute sign of psi
    result = []
    for orbitals in up_down_orbitals:
        signs, ld = np.linalg.slogdet(orbitals)
        signs, ld = signs.reshape(n_scfs, -1), ld.reshape(n_scfs, -1)
        dets = np.exp(ld)
        dets /= dets.sum(-1, keepdims=True)
        base = signs[0]
        result.append(np.array([(np.vdot(base == s, d) >= 0.5)*2-1 for s,
                      d in zip(signs, dets)]).reshape(*orbitals.shape[:-3]))
    return result


class AtomicOrbitals(nn.Module):

	def __init__(self, mol, cuda=False):
		"""Computes the value of atomic orbitals

		Args:
			mol (Molecule): Molecule object
			cuda (bool, optional): Turn GPU ON/OFF Defaults to False.
		"""

		super(AtomicOrbitals, self).__init__()
		dtype = torch.get_default_dtype()

		# wavefunction data
		self.nelec = mol.nelec
		self.norb = mol.basis.nao
		self.ndim = 3

		# make the atomic position optmizable
		self.atom_coords = nn.Parameter(torch.as_tensor(
			mol.basis.atom_coords_internal).type(dtype))
		self.atom_coords.requires_grad = True
		self.natoms = len(self.atom_coords)
		self.atomic_number = mol.atomic_number

		# define the BAS positions.
		self.nshells = torch.as_tensor(mol.basis.nshells)
		self.nao_per_atom = torch.as_tensor(mol.basis.nao_per_atom)
		self.bas_coords = self.atom_coords.repeat_interleave(
			self.nshells, dim=0)
		self.nbas = len(self.bas_coords)

		# index for the contractions
		self.index_ctr = torch.as_tensor(mol.basis.index_ctr)
		self.nctr_per_ao = torch.as_tensor(mol.basis.nctr_per_ao)
		self.contract = not len(torch.unique(
			self.index_ctr)) == len(self.index_ctr)

		# get the coeffs of the bas
		self.bas_coeffs = torch.as_tensor(
			mol.basis.bas_coeffs).type(dtype)

		# get the exponents of the bas
		self.bas_exp = nn.Parameter(
			torch.as_tensor(mol.basis.bas_exp).type(dtype))
		self.bas_exp.requires_grad = True

		# harmonics generator
		self.harmonics_type = mol.basis.harmonics_type
		if mol.basis.harmonics_type == 'sph':
			self.bas_n = torch.as_tensor(mol.basis.bas_n).type(dtype)
			self.harmonics = Harmonics(
				mol.basis.harmonics_type,
				bas_l=mol.basis.bas_l,
				bas_m=mol.basis.bas_m,
				cuda=cuda)

		elif mol.basis.harmonics_type == 'cart':
			self.bas_n = torch.as_tensor(mol.basis.bas_kr).type(dtype)
			self.harmonics = Harmonics(
				mol.basis.harmonics_type,
				bas_kx=mol.basis.bas_kx,
				bas_ky=mol.basis.bas_ky,
				bas_kz=mol.basis.bas_kz,
				cuda=cuda)

		# select the radial apart
		radial_dict = {'sto': radial_slater,
					   'gto': radial_gaussian,
					   'sto_pure': radial_slater_pure,
					   'gto_pure': radial_gaussian_pure}
		self.radial = radial_dict[mol.basis.radial_type]
		self.radial_type = mol.basis.radial_type

		# get the normalisation constants
		if hasattr(mol.basis, 'bas_norm') and False:
			self.norm_cst = torch.as_tensor(
				mol.basis.bas_norm).type(dtype)
		else:
			with torch.no_grad():
				self.norm_cst = atomic_orbital_norm(
					mol.basis).type(dtype)

		self.cuda = cuda
		self.device = torch.device('cpu')
		if self.cuda:
			self._to_device()

	def __repr__(self):
		name = self.__class__.__name__
		return name + '(%s, %s, %d -> (%d,%d) )' % (self.radial_type, self.harmonics_type,
													self.nelec*self.ndim, self.nelec,
													self.norb)

	def _to_device(self):
		"""Export the non parameter variable to the device."""

		self.device = torch.device('cuda')
		self.to(self.device)
		attrs = ['bas_n', 'bas_coeffs',
				 'nshells', 'norm_cst',
				 'index_ctr', 'nctr_per_ao',
				 'nao_per_atom']
		for at in attrs:
			self.__dict__[at] = self.__dict__[at].to(self.device)

	def forward(self, pos, derivative=[0], sum_grad=True, sum_hess=True, one_elec=False):
		"""Computes the values of the atomic orbitals.

		.. math::
			\phi_i(r_j) = \sum_n c_n \\text{Rad}^{i}_n(r_j) \\text{Y}^{i}_n(r_j)

		where Rad is the radial part and Y the spherical harmonics part.
		It is also possible to compute the first and second derivatives

		.. math::
			\\nabla \phi_i(r_j) = \\frac{d}{dx_j} \phi_i(r_j) + \\frac{d}{dy_j} \phi_i(r_j) + \\frac{d}{dz_j} \phi_i(r_j) \n
			\\text{grad} \phi_i(r_j) = (\\frac{d}{dx_j} \phi_i(r_j), \\frac{d}{dy_j} \phi_i(r_j), \\frac{d}{dz_j} \phi_i(r_j)) \n
			\Delta \phi_i(r_j) = \\frac{d^2}{dx^2_j} \phi_i(r_j) + \\frac{d^2}{dy^2_j} \phi_i(r_j) + \\frac{d^2}{dz^2_j} \phi_i(r_j)

		Args:
			pos (torch.tensor): Positions of the electrons
								  Size : Nbatch, Nelec x Ndim
			derivative (int, optional): order of the derivative (0,1,2,).
										Defaults to 0.
			sum_grad (bool, optional): Return the sum_grad (i.e. the sum of
									   the derivatives) or the individual
									   terms. Defaults to True.
									   False only for derivative=1

			sum_hess (bool, optional): Return the sum_hess (i.e. the sum of
									   2nd the derivatives) or the individual
									   terms. Defaults to True.
									   False only for derivative=1

			one_elec (bool, optional): if only one electron is in input

		Returns:
			torch.tensor: Value of the AO (or their derivatives) \n
						  size : Nbatch, Nelec, Norb (sum_grad = True) \n
						  size : Nbatch, Nelec, Norb, Ndim (sum_grad = False)

		Examples::
			>>> mol = Molecule('h2.xyz')
			>>> ao = AtomicOrbitals(mol)
			>>> pos = torch.rand(100,6)
			>>> aovals = ao(pos)
			>>> daovals = ao(pos,derivative=1)
		"""

		if not isinstance(derivative, list):
			derivative = [derivative]

		if not sum_grad:
			assert(1 in derivative)

		if not sum_hess:
			assert(2 in derivative)

		if one_elec:
			nelec_save = self.nelec
			self.nelec = 1

		if derivative == [0]:
			ao = self._compute_ao_values(pos)

		elif derivative == [1]:
			ao = self._compute_first_derivative_ao_values(
				pos, sum_grad)

		elif derivative == [2]:
			ao = self._compute_second_derivative_ao_values(
				pos, sum_hess)

		elif derivative == [3]:
			ao = self._compute_mixed_second_derivative_ao_values(pos)

		elif derivative == [0, 1, 2]:
			ao = self._compute_all_ao_values(pos)

		else:
			raise ValueError(
				'derivative must be 0, 1, 2, 3 or [0, 1, 2, 3], got ', derivative)

		if one_elec:
			self.nelec = nelec_save

		return ao

	def _compute_ao_values(self, pos):
		"""Compute the value of the ao from the xyx and r tensor

		Args:
			pos (torch.tensor): position of each elec size Nbatch, NelexNdim

		Returns:
			torch.tensor: atomic orbital values size (Nbatch, Nelec, Norb)
		"""
		xyz, r = self._process_position(pos)

		R = self.radial(r, self.bas_n, self.bas_exp)

		Y = self.harmonics(xyz)
		return self._ao_kernel(R, Y)

	def _ao_kernel(self, R, Y):
		"""Kernel for the ao values

		Args:
			R (torch.tensor): radial part of the AOs
			Y (torch.tensor): harmonics part of the AOs

		Returns:
			torch.tensor: values of the AOs (with contraction)
		"""
		ao = self.norm_cst * R * Y
		if self.contract:
			ao = self._contract(ao)
		return ao

	def _compute_first_derivative_ao_values(self, pos, sum_grad):
		"""Compute the value of the derivative of the ao from the xyx and r tensor

		Args:
			pos (torch.tensor): position of each elec size Nbatch, Nelec x Ndim
			sum_grad (boolean): return the sum_grad (True) or gradient (False)

		Returns:
			torch.tensor: derivative of atomic orbital values
						  size (Nbatch, Nelec, Norb) if sum_grad
						  size (Nbatch, Nelec, Norb, Ndim) if sum_grad=False
		"""
		if sum_grad:
			return self._compute_sum_gradient_ao_values(pos)
		else:
			return self._compute_gradient_ao_values(pos)

	def _compute_sum_gradient_ao_values(self, pos):
		"""Compute the jacobian of the ao from the xyx and r tensor

		Args:
			pos (torch.tensor): position of each elec size Nbatch, Nelec x Ndim

		Returns:
			torch.tensor: derivative of atomic orbital values
						  size (Nbatch, Nelec, Norb)

		"""

		xyz, r = self._process_position(pos)

		R, dR = self.radial(r, self.bas_n,
							self.bas_exp, xyz=xyz,
							derivative=[0, 1])

		Y, dY = self.harmonics(xyz, derivative=[0, 1])

		return self._sum_gradient_kernel(R, dR, Y, dY)

	def _sum_gradient_kernel(self, R, dR, Y, dY):
		"""Kernel for the jacobian of the ao values

		Args:
			R (torch.tensor): radial part of the AOs
			dR (torch.tensor): derivative of the radial part of the AOs
			Y (torch.tensor): harmonics part of the AOs
			dY (torch.tensor): derivative of the harmonics part of the AOs

		Returns:
			torch.tensor: values of the jacobian of the AOs (with contraction)
		"""
		dao = self.norm_cst * (dR * Y + R * dY)
		if self.contract:
			dao = self._contract(dao)
		return dao

	def _compute_gradient_ao_values(self, pos):
		"""Compute the gradient of the ao from the xyx and r tensor

		Args:
			pos (torch.tensor): position of each elec size Nbatch, Nelec x Ndim

		Returns:
			torch.tensor: derivative of atomic orbital values
						  size (Nbatch, Nelec, Norb, Ndim)

		"""
		xyz, r = self._process_position(pos)

		R, dR = self.radial(r, self.bas_n,
							self.bas_exp, xyz=xyz,
							derivative=[0, 1],
							sum_grad=False)

		Y, dY = self.harmonics(xyz, derivative=[0, 1], sum_grad=False)

		return self._gradient_kernel(R, dR, Y, dY)

	def _gradient_kernel(self, R, dR, Y, dY):
		"""Kernel for the gradient of the ao values

		Args:
			R (torch.tensor): radial part of the AOs
			dR (torch.tensor): derivative of the radial part of the AOs
			Y (torch.tensor): harmonics part of the AOs
			dY (torch.tensor): derivative of the harmonics part of the AOs

		Returns:
			torch.tensor: values of the gradient of the AOs (with contraction)
		"""
		nbatch = R.shape[0]
		bas = dR * Y.unsqueeze(-1) + R.unsqueeze(-1) * dY

		bas = self.norm_cst.unsqueeze(-1) * \
			self.bas_coeffs.unsqueeze(-1) * bas

		if self.contract:
			ao = torch.zeros(nbatch, self.nelec, self.norb,
							 3, device=self.device).type(torch.get_default_dtype())
			ao.index_add_(2, self.index_ctr, bas)
		else:
			ao = bas
		return ao

	def _compute_second_derivative_ao_values(self, pos, sum_hess):
		"""Compute the values of the 2nd derivative of the ao from the xyz and r tensors

		Args:
			pos (torch.tensor): position of each elec size Nbatch, Nelec x Ndim
			sum_hess (boolean): return the sum_hess (True) or gradient (False)

		Returns:
			torch.tensor: derivative of atomic orbital values
						  size (Nbatch, Nelec, Norb) if sum_hess
						  size (Nbatch, Nelec, Norb, Ndim) if sum_hess=False
		"""
		if sum_hess:
			return self._compute_sum_diag_hessian_ao_values(pos)
		else:
			return self._compute_diag_hessian_ao_values(pos)

	def _compute_sum_diag_hessian_ao_values(self, pos):
		"""Compute the laplacian of the ao from the xyx and r tensor

		Args:
			pos (torch.tensor): position of each elec size Nbatch, Nelec x Ndim

		Returns:
			torch.tensor: derivative of atomic orbital values
						  size (Nbatch, Nelec, Norb)

		"""
		xyz, r = self._process_position(pos)

		R, dR, d2R = self.radial(r, self.bas_n, self.bas_exp,
								 xyz=xyz, derivative=[0, 1, 2],
								 sum_grad=False)

		Y, dY, d2Y = self.harmonics(xyz,
									derivative=[0, 1, 2],
									sum_grad=False)
		return self._sum_diag_hessian_kernel(R, dR, d2R, Y, dY, d2Y)

	def _sum_diag_hessian_kernel(self, R, dR, d2R, Y, dY, d2Y):
		"""Kernel for the sum of the diag hessian of the ao values

		Args:
			R (torch.tensor): radial part of the AOs
			dR (torch.tensor): derivative of the radial part of the AOs
			d2R (torch.tensor): 2nd derivative of the radial part of the AOs
			Y (torch.tensor): harmonics part of the AOs
			dY (torch.tensor): derivative of the harmonics part of the AOs
			d2Y (torch.tensor): 2nd derivative of the harmonics part of the AOs

		Returns:
			torch.tensor: values of the laplacian of the AOs (with contraction)
		"""

		d2ao = self.norm_cst * \
			(d2R * Y + 2. * (dR * dY).sum(3) + R * d2Y)
		if self.contract:
			d2ao = self._contract(d2ao)
		return d2ao

	def _compute_diag_hessian_ao_values(self, pos):
		"""Compute the individual elements of the laplacian of the ao from the xyx and r tensor

		Args:
			pos (torch.tensor): position of each elec size Nbatch, Nelec x Ndim

		Returns:
			torch.tensor: derivative of atomic orbital values
						  size (Nbatch, Nelec, Norb, 3)

		"""

		xyz, r = self._process_position(pos)

		R, dR, d2R = self.radial(r, self.bas_n, self.bas_exp,
								 xyz=xyz, derivative=[0, 1, 2],
								 sum_grad=False, sum_hess=False)

		Y, dY, d2Y = self.harmonics(xyz,
									derivative=[0, 1, 2],
									sum_grad=False, sum_hess=False)

		return self._diag_hessian_kernel(R, dR, d2R, Y, dY, d2Y)

	def _diag_hessian_kernel(self, R, dR, d2R, Y, dY, d2Y):
		"""Kernel for the diagonal hessian of the ao values

		Args:
			R (torch.tensor): radial part of the AOs
			dR (torch.tensor): derivative of the radial part of the AOs
			d2R (torch.tensor): 2nd derivative of the radial part of the AOs
			Y (torch.tensor): harmonics part of the AOs
			dY (torch.tensor): derivative of the harmonics part of the AOs
			d2Y (torch.tensor): 2nd derivative of the harmonics part of the AOs

		Returns:
			torch.tensor: values of the laplacian of the AOs (with contraction)
		"""

		nbatch = R.shape[0]

		bas = self.norm_cst.unsqueeze(-1) * self.bas_coeffs.unsqueeze(-1) * \
			(d2R * Y.unsqueeze(-1) + 2. *
			 (dR * dY) + R.unsqueeze(-1) * d2Y)

		if self.contract:
			d2ao = torch.zeros(nbatch, self.nelec, self.norb,
							   3, device=self.device).type(torch.get_default_dtype())
			d2ao.index_add_(2, self.index_ctr, bas)
		else:
			d2ao = bas

		return d2ao

	def _compute_mixed_second_derivative_ao_values(self, pos):
		"""Compute the mixed second derivative of the ao from the xyx and r tensor

		Args:
			pos (torch.tensor): position of each elec size Nbatch, Nelec x Ndim

		Returns:
			torch.tensor: derivative of atomic orbital values
						  size (Nbatch, Nelec, Norb)

		"""
		xyz, r = self._process_position(pos)

		R, dR, d2R, d2mR = self.radial(r, self.bas_n, self.bas_exp,
									   xyz=xyz, derivative=[
										   0, 1, 2, 3],
									   sum_grad=False)

		Y, dY, d2Y, d2mY = self.harmonics(xyz,
										  derivative=[0, 1, 2, 3],
										  sum_grad=False)

		return self._off_diag_hessian_kernel(R, dR, d2R, d2mR, Y, dY, d2Y, d2mY)

	def _off_diag_hessian_kernel(self, R, dR, d2R, d2mR, Y, dY, d2Y, d2mY):
		"""Kernel for the off diagonal hessian of the ao values

		Args:
			R (torch.tensor): radial part of the AOs
			dR (torch.tensor): derivative of the radial part of the AOs
			d2R (torch.tensor): 2nd derivative of the radial part of the AOs
			d2mR (torch.tensor): mixed 2nd derivative of the radial part of the AOs
			Y (torch.tensor): harmonics part of the AOs
			dY (torch.tensor): derivative of the harmonics part of the AOs
			d2Y (torch.tensor): 2nd derivative of the harmonics part of the AOs
			d2mY (torch.tensor): 2nd mixed derivative of the harmonics part of the AOs

		Returns:
			torch.tensor: values of the mixed derivative of the AOs (with contraction)
		"""

		nbatch = R.shape[0]

		bas = self.norm_cst.unsqueeze(-1) * self.bas_coeffs.unsqueeze(-1) * \
			(d2mR * Y.unsqueeze(-1) +
			 ((dR[..., [[0, 1], [0, 2], [1, 2]]] *
			   dY[..., [[1, 0], [2, 0], [2, 1]]]).sum(-1))
			 + R.unsqueeze(-1) * d2mY)

		if self.contract:
			d2ao = torch.zeros(nbatch, self.nelec, self.norb,
							   3, device=self.device).type(torch.get_default_dtype())
			d2ao.index_add_(2, self.index_ctr, bas)
		else:
			d2ao = bas

		return d2ao

	def _compute_all_ao_values(self, pos):
		"""Compute the ao, gradient, laplacian of the ao from the xyx and r tensor

		Args:
			pos (torch.tensor): position of each elec size Nbatch, Nelec x Ndim
			sum_grad (bool): return the sum of the gradients if True
			sum_hess (bool): returns the sum of the diag hess if True
		Returns:
			tuple(): (ao, grad and lapalcian) of atomic orbital values
					 ao size (Nbatch, Nelec, Norb)
					 dao size (Nbatch, Nelec, Norb, Ndim)
					 d2ao size (Nbatch, Nelec, Norb)

		"""

		xyz, r = self._process_position(pos)

		# the gradients elements are needed to compute the second der
		# we therefore use sum_grad=False regardless of the input arg
		R, dR, d2R = self.radial(r, self.bas_n, self.bas_exp,
								 xyz=xyz, derivative=[0, 1, 2],
								 sum_grad=False)

		# the gradients elements are needed to compute the second der
		# we therefore use sum_grad=False regardless of the input arg
		Y, dY, d2Y = self.harmonics(xyz,
									derivative=[0, 1, 2],
									sum_grad=False)

		return (self._ao_kernel(R, Y),
				self._gradient_kernel(R, dR, Y, dY),
				self._sum_diag_hessian_kernel(R, dR, d2R, Y, dY, d2Y))

	def _process_position(self, pos):
		"""Computes the positions/distance bewteen elec/orb

		Args:
			pos (torch.tensor): positions of the walkers Nbat, NelecxNdim

		Returns:
			torch.tensor, torch.tensor: positions of the elec wrt the bas
										(Nbatch, Nelec, Norb, Ndim)
										distance between elec and bas
										(Nbatch, Nelec, Norb)
		"""
		# get the elec-atom vectors/distances
		xyz, r = self._elec_atom_dist(pos)

		# repeat/interleave to get vector and distance between
		# electrons and orbitals
		return (xyz.repeat_interleave(self.nshells, dim=2),
				r.repeat_interleave(self.nshells, dim=2))

	def _elec_atom_dist(self, pos):
		"""Computes the positions/distance bewteen elec/atoms

		Args:
			pos (torch.tensor): positions of the walkers : Nbatch x [Nelec*Ndim]

		Returns:
			(torch.tensor, torch.tensor): positions of the elec wrt the atoms
										[Nbatch x Nelec x Natom x Ndim]
										distance between elec and atoms
										[Nbatch x Nelec x Natom]
		"""

		# compute the vectors between electrons and atoms
		xyz = (pos.view(-1, self.nelec, 1, self.ndim) -
			   self.atom_coords[None, ...])

		# distance between electrons and atoms
		r = torch.sqrt((xyz*xyz).sum(3))

		return xyz, r

	def _contract(self, bas):
		"""Contrat the basis set to form the atomic orbitals

		Args:
			bas (torch.tensor): values of the basis function

		Returns:
			torch.tensor: values of the contraction
		"""
		nbatch = bas.shape[0]
		bas = self.bas_coeffs * bas
		cbas = torch.zeros(nbatch, self.nelec,
						   self.norb, device=self.device
						   ).type(torch.get_default_dtype())
		cbas.index_add_(2, self.index_ctr, bas)
		return cbas

	def update(self, ao, pos, idelec):
		"""Update an AO matrix with the new positions of one electron

		Args:
			ao (torch.tensor): initial AO matrix
			pos (torch.tensor): new positions of some electrons
			idelec (int): index of the electron that has moved

		Returns:
			torch.tensor: new AO matrix

		Examples::
			>>> mol = Molecule('h2.xyz')
			>>> ao = AtomicOrbitals(mol)
			>>> pos = torch.rand(100,6)
			>>> aovals = ao(pos)
			>>> id = 0
			>>> pos[:,:3] = torch.rand(100,3)
			>>> ao.update(aovals, pos, 0)
		"""

		ao_new = ao.clone()
		ids, ide = (idelec) * 3, (idelec + 1) * 3
		ao_new[:, idelec, :] = self.forward(
			pos[:, ids:ide], one_elec=True).squeeze(1)
		return ao_new


class SlaterJastrowBase(WaveFunction):

	def __init__(self, mol,
					configs='ground_state',
					kinetic='jacobi',
					cuda=False,
					include_all_mo=True):
		"""Implementation of the QMC Network.

		Args:
			mol (qmc.wavefunction.Molecule): a molecule object
			configs (str, optional): defines the CI configurations to be used. Defaults to 'ground_state'.
			kinetic (str, optional): method to compute the kinetic energy. Defaults to 'jacobi'.
			use_jastrow (bool, optional): turn jastrow factor ON/OFF. Defaults to True.
			cuda (bool, optional): turns GPU ON/OFF  Defaults to False.
			include_all_mo (bool, optional): include either all molecular orbitals or only the ones that are
												popualted in the configs. Defaults to False
		Examples::
			>>> mol = Molecule('h2o.xyz', calculator='adf', basis = 'dzp')
			>>> wf = SlaterJastrow(mol, configs='cas(2,2)')
		"""

		super(SlaterJastrowBase, self).__init__(
			mol.nelec, 3, kinetic, cuda)

		# check for cuda
		if not torch.cuda.is_available and self.cuda:
			raise ValueError('Cuda not available, use cuda=False')

		# check for conf/mo size
		if not include_all_mo and configs.startswith('cas('):
			raise ValueError(
				'CAS calculation only possible with include_all_mo=True')

		# number of atoms
		self.mol = mol
		self.atoms = mol.atoms
		self.natom = mol.natom

		# define the SD we want
		self.orb_confs = OrbitalConfigurations(mol)
		self.configs_method = configs
		self.configs = self.orb_confs.get_configs(configs)
		self.nci = len(self.configs[0])
		self.highest_occ_mo = torch.stack(self.configs).max()+1

		# define the atomic orbital layer
		self.ao = AtomicOrbitals(mol, cuda)

		# define the mo layer
		self.include_all_mo = include_all_mo
		self.nmo_opt = mol.basis.nmo if include_all_mo else self.highest_occ_mo
		self.mo_scf = nn.Linear(
			mol.basis.nao, self.nmo_opt, bias=False)
		self.mo_scf.weight = self.get_mo_coeffs()
		self.mo_scf.weight.requires_grad = False
		if self.cuda:
			self.mo_scf.to(self.device)

		# define the mo mixing layer
		# self.mo = nn.Linear(mol.basis.nmo, self.nmo_opt, bias=False)
		self.mo = nn.Linear(self.nmo_opt, self.nmo_opt, bias=False)
		self.mo.weight = nn.Parameter(
			torch.eye(self.nmo_opt, self.nmo_opt))
		if self.cuda:
			self.mo.to(self.device)

		# jastrow
		self.jastrow_type = None
		self.use_jastrow = False

		#  define the SD pooling layer
		self.pool = SlaterPooling(self.configs_method,
									self.configs, mol, cuda)

		# define the linear layer
		self.fc = nn.Linear(self.nci, 1, bias=False)
		self.fc.weight.data.fill_(0.)
		self.fc.weight.data[0][0] = 1.

		if self.cuda:
			self.fc = self.fc.to(self.device)

		self.kinetic_method = kinetic
		if kinetic == 'jacobi':
			self.kinetic_energy = self.kinetic_energy_jacobi

		gradients = 'auto'
		self.gradients_method = gradients
		if gradients == 'jacobi':
			self.gradients = self.gradients_jacobi

		if self.cuda:
			self.device = torch.device('cuda')
			self.to(self.device)

		# register the callable for hdf5 dump
		register_extra_attributes(self,
									['ao', 'mo_scf',
									'mo', 'jastrow',
									'pool', 'fc'])

		def log_data(self):
		"""Print information abut the wave function."""
		log.info('')
		log.info(' Wave Function')
		log.info('  Jastrow factor      : {0}', self.use_jastrow)
		if self.use_jastrow:
			log.info(
				'  Jastrow kernel      : {0}', self.jastrow_type)
		log.info('  Highest MO included : {0}', self.nmo_opt)
		log.info('  Configurations      : {0}', self.configs_method)
		log.info('  Number of confs     : {0}', self.nci)

		log.debug('  Configurations      : ')
		for ic in range(self.nci):
			cstr = ' ' + ' '.join([str(i)
									for i in self.configs[0][ic].tolist()])
			cstr += ' | ' + ' '.join([str(i)
										for i in self.configs[1][ic].tolist()])
			log.debug(cstr)

		log.info('  Kinetic energy      : {0}', self.kinetic_method)
		log.info(
			'  Number var  param   : {0}', self.get_number_parameters())
		log.info('  Cuda support        : {0}', self.cuda)
		if self.cuda:
			log.info(
				'  GPU                 : {0}', torch.cuda.get_device_name(0))


		def get_mo_coeffs(self):
		mo_coeff = torch.as_tensor(self.mol.basis.mos).type(
			torch.get_default_dtype())
		if not self.include_all_mo:
			mo_coeff = mo_coeff[:, :self.highest_occ_mo]
		return nn.Parameter(mo_coeff.transpose(0, 1).contiguous())


		def update_mo_coeffs(self):
		self.mol.atom_coords = self.ao.atom_coords.detach().numpy().tolist()
		self.mo.weight = self.get_mo_coeffs()


		def geometry(self, pos):
		"""Returns the gemoetry of the system in xyz format

		Args:
			pos (torch.tensor): sampling points (Nbatch, 3*Nelec)

		Returns:
			list: list where each element is one line of the xyz file
		"""
		d = []
		for iat in range(self.natom):

			xyz = self.ao.atom_coords[iat,
										:].cpu().detach().numpy().tolist()
			d.append(xyz)
		return d


		def gto2sto(self, plot=False):
		"""Fits the AO GTO to AO STO.
			The SZ sto that have only one basis function per ao
		"""

		assert(self.ao.radial_type.startswith('gto'))
		assert(self.ao.harmonics_type == 'cart')

		log.info('  Fit GTOs to STOs  : ')

		def sto(x, norm, alpha):
			"""Fitting function."""
			return norm * np.exp(-alpha * np.abs(x))

		# shortcut for nao
		nao = self.mol.basis.nao

		# create a new mol and a new basis
		new_mol = deepcopy(self.mol)
		basis = deepcopy(self.mol.basis)

		# change basis to sto
		basis.radial_type = 'sto_pure'
		basis.nshells = self.ao.nao_per_atom.detach().cpu().numpy()

		# reset basis data
		basis.index_ctr = np.arange(nao)
		basis.bas_coeffs = np.ones(nao)
		basis.bas_exp = np.zeros(nao)
		basis.bas_norm = np.zeros(nao)
		basis.bas_kr = np.zeros(nao)
		basis.bas_kx = np.zeros(nao)
		basis.bas_ky = np.zeros(nao)
		basis.bas_kz = np.zeros(nao)

		# 2D fit space
		x = torch.linspace(-5, 5, 501)

		# compute the values of the current AOs using GTO BAS
		pos = x.reshape(-1, 1).repeat(1, self.ao.nbas).to(self.device)
		gto = self.ao.norm_cst * torch.exp(-self.ao.bas_exp*pos**2)
		gto = gto.unsqueeze(1).repeat(1, self.nelec, 1)
		ao = self.ao._contract(gto)[
			:, 0, :].detach().cpu().numpy()

		# loop over AOs
		for iorb in range(self.ao.norb):

			# fit AO with STO
			xdata = x.numpy()
			ydata = ao[:, iorb]
			popt, pcov = curve_fit(sto, xdata, ydata)

			# store new exp/norm
			basis.bas_norm[iorb] = popt[0]
			basis.bas_exp[iorb] = popt[1]

			# determine k values
			basis.bas_kx[iorb] = self.ao.harmonics.bas_kx[self.ao.index_ctr == iorb].unique(
			).item()
			basis.bas_ky[iorb] = self.ao.harmonics.bas_ky[self.ao.index_ctr == iorb].unique(
			).item()
			basis.bas_kz[iorb] = self.ao.harmonics.bas_kz[self.ao.index_ctr == iorb].unique(
			).item()

			# plot if necessary
			if plot:
				plt.plot(xdata, ydata)
				plt.plot(xdata, sto(xdata, *popt))
				plt.show()

		# update basis in new mole
		new_mol.basis = basis

		# returns new orbital instance
		return self.__class__(new_mol, configs=self.configs_method,
								kinetic=self.kinetic_method,
								cuda=self.cuda,
								include_all_mo=self.include_all_mo)