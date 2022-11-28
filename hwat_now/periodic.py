def trs_v_space(x, basis)->jnp.array:
    """ basis: 3 *row* vectors
    intuition: points need info from all basis vectors 
    (m, n) (n, n) -> (m, n) """
    return jnp.dot(x, basis)

def apply_torus(x, basis, basis_inv):
    x_dash = trs_v_space(x, basis_inv)
    x_dash = jnp.fmod(x_dash, 1.)
    x_dash = jnp.where(x_dash < 0., x_dash + 1., x_dash)
    x_dash = trs_v_space(x_dash, basis)
    return x_dash


def apply_minimum_image_convention(displacement_vectors, basis, inv_basis, on=False):
    '''
    pseudocode:
        - translate to the unit cell 
        - compute the distances
        - 2 * element distances will be maximum 0.999 (as always in the same cell)
        - int(2 * element distances) will either be 0, 1 or -1
        # displacement_vectors = displacement_vectors - lax.stop_gradient(displace)  #
    '''
    displace = (2. * transform_vector_space(displacement_vectors, inv_basis, on=on)).astype(int).astype(displacement_vectors.dtype)
    displace = transform_vector_space(displace, basis, on=on)
    displacement_vectors = displacement_vectors - displace
    return displacement_vectors



def create_energy_fn(mol, vwf, separate=False):
    ### Remember the normalisation per electron / atoms / whatever 
    local_kinetic_energy = create_local_kinetic_energy(vwf)
    compute_potential_energy = create_potential_energy(mol)

    def _compute_local_energy(params, walkers):
        potential_energy = compute_potential_energy(walkers, mol.r_atoms, mol.z_atoms)
        kinetic_energy = local_kinetic_energy(params, walkers)
        if mol.pbc:
            potential_energy /= mol.n_atoms if not mol.n_atoms == 0 else mol.n_el
            kinetic_energy /= mol.n_atoms if not mol.n_atoms == 0 else mol.n_el
        if mol.system == 'HEG':
            potential_energy /= mol.density_parameter
            kinetic_energy /= mol.density_parameter**2
        if separate:
            return potential_energy, kinetic_energy
        else:
            return potential_energy + kinetic_energy
        
    return _compute_local_energy