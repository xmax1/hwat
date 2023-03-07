

"""
This file contains two classes: Atom and Molecule.
An atom consists of an element and coordinates while a molecule
is composed by a set of atoms.
The classes contain simple logic functions to obtain spins, charges
and coordinates for molecules.
"""
import math
import numbers
from typing import Counter, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
from pesnet.constants import ANGSTROM_TO_BOHR
from pesnet.systems import ELEMENT_BY_ATOMIC_NUM, ELEMENT_BY_SYMBOL
from pyscf import gto



from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
from torch import nn

import torch
from .. import log
from ..utils import register_extra_attributes
from .orbitals.atomic_orbitals import AtomicOrbitals
from .pooling.orbital_configurations import OrbitalConfigurations
from .pooling.slater_pooling import SlaterPooling
from .wf_base import WaveFunction




""" notes
pyscf:issues
Extension modules are not available on the Conda cloud. 
They should be installed either with pip, or through the environment variable PYSCF_EXT_PATH (see the section Extension modules).
https://pyscf.org/install.html

pyscf:examples
https://mattermodeling.stackexchange.com/questions/9410/molecular-orbital-values-on-grid-points-in-pyscf

"""

from typing import Optional, Tuple

import numpy as np
import pyscf


class Scf:
    """
    A wrapper class around PyScf's Scf. The main benefit of this class is that
    it enables one to easily obtain the molecular orbitals for a given set of 
    electrons.
    """

    def __init__(self, mol: pyscf.gto.Mole, restricted: bool = True) -> None:
        self.mol = mol
        self.restricted = restricted

    def run(self, initial_guess: Optional['Scf'] = None):
        if self.restricted:
            self._mean_field = pyscf.scf.RHF(self.mol)
        else:
            self._mean_field = pyscf.scf.UHF(self.mol)
        if initial_guess is None:
            self._mean_field.kernel()
        else:
            self._mean_field.kernel(initial_guess._mean_field.make_rdm1())
        return self._mean_field

    def eval_molecular_orbitals(self, electrons: np.ndarray, deriv: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        
        if self.restricted:
            coeffs = (self._mean_field.mo_coeff,)
        else:
            coeffs = self._mean_field.mo_coeff

        if self.mol.cart:
            raise NotImplementedError(
                'Evaluation of molecular orbitals using cartesian GTOs.')

        gto_op = 'GTOval_sph_deriv1' if deriv else 'GTOval_sph'
        ao_values = self.mol.eval_gto(gto_op, electrons)
        mo_values = tuple(np.matmul(ao_values, coeff) for coeff in coeffs)
        if self.restricted:
            mo_values *= 2
        return mo_values





[docs]class SlaterJastrowBase(WaveFunction):

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

[docs]    def log_data(self):
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


[docs]    def get_mo_coeffs(self):
        mo_coeff = torch.as_tensor(self.mol.basis.mos).type(
            torch.get_default_dtype())
        if not self.include_all_mo:
            mo_coeff = mo_coeff[:, :self.highest_occ_mo]
        return nn.Parameter(mo_coeff.transpose(0, 1).contiguous())


[docs]    def update_mo_coeffs(self):
        self.mol.atom_coords = self.ao.atom_coords.detach().numpy().tolist()
        self.mo.weight = self.get_mo_coeffs()


[docs]    def geometry(self, pos):
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


[docs]    def gto2sto(self, plot=False):
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


class Atom:
    def __init__(self, name: str, coords: Optional[Sequence[float]] = None, units='bohr') -> None:
        assert units in ['bohr', 'angstrom']
        if isinstance(name, str):
            self.element = ELEMENT_BY_SYMBOL[name]
        elif isinstance(name, numbers.Number):
            self.element = ELEMENT_BY_ATOMIC_NUM[name]
        else:
            raise ValueError()
        self.coords = coords
        if self.coords is None:
            self.coords = (0, 0, 0)
        assert len(self.coords) == 3
        self.coords = np.array(coords)
        if units == 'angstrom':
            self.coords *= ANGSTROM_TO_BOHR

    @property
    def atomic_number(self):
        return self.element.atomic_number

    @property
    def symbol(self):
        return self.element.symbol

    def __str__(self) -> str:
        return self.element.symbol


class Molecule:
    def __init__(self, atoms: Sequence[Atom], spins: Optional[Tuple[int, int]] = None) -> None:
        self.atoms = atoms
        self._spins = spins

    def charges(self):
        return tuple([a.atomic_number for a in self.atoms])

    def coords(self):
        coords = jnp.array([a.coords for a in self.atoms])
        coords -= coords.mean(0, keepdims=True)
        return coords

    def spins(self):
        if self._spins is not None:
            return self._spins
        else:
            n_electrons = sum(self.charges())
            return (math.ceil(n_electrons/2), math.floor(n_electrons/2))

    def to_pyscf(self, basis='STO-6G', verbose: int = 3):
        mol = gto.Mole(atom=[
            [a.symbol, coords]
            for a, coords in zip(self.atoms, self.coords())
        ], basis=basis, unit='bohr', verbose=verbose)
        spins = self.spins()
        mol.spin = spins[0] - spins[1]
        nuc_charge = sum(a.atomic_number for a in self.atoms)
        e_charge = sum(spins)
        mol.charge = nuc_charge - e_charge
        mol.build()
        return mol

    def __str__(self) -> str:
        result = ''
        if len(self.atoms) == 1:
            result = str(self.atoms[0])
        elif len(self.atoms) == 2:
            distance = np.linalg.norm(
                self.atoms[0].coords - self.atoms[1].coords)
            result = f'{str(self.atoms[0])}-{str(self.atoms[1])}_{distance:.2f}'
        else:
            vals = dict(Counter(str(a) for a in self.atoms))
            result = ''.join(f'{key}{val}' for key, val in vals.items())
        if sum(self.spins()) < sum(self.charges()):
            result += 'plus'
        elif sum(self.spins()) > sum(self.charges()):
            result += 'minus'
        return result