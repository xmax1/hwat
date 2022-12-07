


@chex.dataclass
class AuxiliaryLossData:
    variance: jnp.DeviceArray
    local_energy: jnp.DeviceArray
    imaginary: jnp.DeviceArray
    kinetic: jnp.DeviceArray
    ewald: jnp.DeviceArray




def make_loss(network, batch_network,
              simulation_cell,
              clip_local_energy=5.0,
              clip_type='real',
              mode='for',
              partition_number=3):
    """
    generates loss function used for wavefunction trains.
    :param network: unbatched logdet function of wavefunction
    :param batch_network: batched logdet function of wavefunction
    :param simulation_cell: pyscf object of simulation cell.
    :param clip_local_energy: clip window width of local energy.
    :param clip_type: specify the clip style. real mode clips the local energy in Cartesion style,
    and complex mode in polar style
    :param mode: specify the evaluation style of local energy.
    'for' mode calculates the laplacian of each electron one by one, which is slow but save GPU memory
    'hessian' mode calculates the laplacian in a highly parallized mode, which is fast but require GPU memory
    'partition' mode calculate the laplacian in a moderate way.
    :param partition_number: Only used if 'partition' mode is employed.
    partition_number must be divisivle by (dim * number of electrons).
    The smaller the faster, but requires more memory.
    :return: the loss function
    """
    el_fun = hamiltonian.local_energy_seperate(network,
                                               simulation_cell=simulation_cell,
                                               mode=mode,
                                               partition_number=partition_number)
    batch_local_energy = jax.vmap(el_fun, in_axes=(None, 0), out_axes=0)

    @jax.custom_jvp
    def total_energy(params, data):
        """
        :param params: a dictionary of parameters
        :param data: batch electron coord with shape [Batch, Nelec * Ndim]
        :return: energy expectation of corresponding walkers (only take real part) with shape [Batch]
        """
        ke, ew = batch_local_energy(params, data)
        e_l = ke + ew
        mean_e_l = jnp.mean(e_l)

        pmean_loss = constants.pmean_if_pmap(mean_e_l, axis_name=constants.PMAP_AXIS_NAME)
        variance = constants.pmean_if_pmap(jnp.mean(jnp.abs(e_l)**2) - jnp.abs(mean_e_l.real) ** 2,
                                           axis_name=constants.PMAP_AXIS_NAME)
        loss = pmean_loss.real
        imaginary = pmean_loss.imag

        return loss, AuxiliaryLossData(variance=variance,
                                       local_energy=e_l,
                                       imaginary=imaginary,
                                       kinetic=ke,
                                       ewald=ew,
                                       )