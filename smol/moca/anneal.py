"""
Implementation of functions for simulated annealing using the ensemble classes
defined in the moca.ensemble module
"""

import numpy as np


def anneal(ensemble, start_temperature, end_temperature, steps, mc_iterations,
           cool_function=None):
    """
    Carries out a simulated annealing procedure for a total number of
    temperatures given by "steps" interpolating between the start and end
    temperature according to a cooling function

    Args:
        ensemble (Ensemble):
            An ensemble class to run Monte Carlo
        start_temperature (float):
            Starting temperature. Must be higher than end temperature.
        end_temperature (float):
            Ending Temperature.
        steps (int):
            Number of temperatures to run MC simulations between start and
            end temperatures.
        mc_iterations (int):
            number of Monte Carlo iterations to run at each temperature.
        cool_function (str):
            A (monotonically decreasing) function to interpolate temperatures.
            If none is given, linear interpolation is used.

    Returns: (minimum energy, occupation)
        tuple
    """
    if start_temperature < end_temperature:
        raise ValueError(f'End temperature is greater than start temperature '
                         f'{end_temperature} > {start_temperature}.')
    if cool_function is None:
        temperatures = np.linspace(start_temperature, end_temperature, steps)
    else:
        raise NotImplementedError('No other cooling functions implented yet.')

    min_energy = ensemble.minimum_energy
    min_occupancy = ensemble.minimum_energy_occupancy

    for T in temperatures:
        ensemble.temperature = T
        ensemble.run(mc_iterations)
        if ensemble.minimum_energy < min_energy:
            min_energy = ensemble.minimum_energy
            min_occupancy = ensemble.minimum_energy_occupancy

    return min_energy, min_occupancy
