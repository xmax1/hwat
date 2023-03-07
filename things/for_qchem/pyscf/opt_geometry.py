from pyscf import gto, scf
from pyscf.geomopt.geometric_solver import optimize
mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='ccpvdz')
mf = scf.RHF(mol)

# geometric
mol_eq = optimize(mf, maxsteps=100)
print(mol_eq.atom_coords())

# pyberny
from pyscf.geomopt.berny_solver import opt_berny
mol_eq = opt_berny(mf, maxsteps=100)
print(mol_eq.atom_coords())