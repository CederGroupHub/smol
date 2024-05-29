"""Tools to characterize MC convergence."""

__author__ = "Ronald Kam"

from warnings import warn

import numpy as np


def check_property_converged(
    property_array, conv_tol=None, min_std=1e-4, last_m=None, verbose=False
):
    """Check if a property is converged in MC run.

    A relatively rudimentary check for convergence, perhaps not the most rigorous.
    The criteria for convergence are:
     1) The last value of the property should be close to the mean of the property, +/-
        some tolerance.

     2) The cumulative mean of the property in the last M samples should be close to
        the mean of the property, +/- some tolerance.

    Args:
        property_array (ndarray):
            array of numerical values of a property (energy, composition, etc)
        conv_tol (float):
            value to determine the convergence tolerance. If None, the std. dev. of the
            property will be used
        min_std (float):
            minimum standard deviation of the property required to perform this
            convergence test. If std dev is too small, then we can assume that there
            were few acceptances and that the simulation is converged.
        last_m (int):
            Number of last M samples to determine the convergence of cumulative property
            mean. If None, take the last 10% of samples.

    Returns:
        converged (bool):
            True if converged, False if not

    """
    property_array = np.array(property_array)
    std_prop = np.std(property_array)
    if std_prop < min_std:
        # Check will never pass if the property std dev is too small, which means there
        # were very few acceptances. Assume in this case that we are already close to
        # the equilibrium value and that MC is converged. This is common for MC runs at
        # low temperature.

        if verbose:
            print(
                "The std dev of the property is very small, so it can be assumed that"
                "MC is converged."
            )
        return True

    if conv_tol is None:
        conv_tol = std_prop

    mean_prop = np.average(property_array)
    n_samples = len(property_array)

    if last_m is None:
        last_m = int(n_samples / 10)

    elif last_m > n_samples:
        warn(
            f"The specified number of last M samples ({last_m}) is greater than the "
            f"number of samples ({n_samples})! Changing to the default value of last "
            f"10% of samples ({int(n_samples / 10)})"
        )
        last_m = int(n_samples / 10)

    # Check criteria 1
    converged_last_val = 0
    if (
        property_array[-1] < mean_prop + conv_tol
        and property_array[-1] > mean_prop - conv_tol
    ):
        converged_last_val = True

    else:
        if verbose:
            print("The last value of the property is not close to the mean.")

    # Check criteria 2
    converged_cum_mean = 0
    prop_cum_mean = np.divide(np.cumsum(property_array), np.arange(1, n_samples + 1))
    if all(prop_cum_mean[-last_m:] < mean_prop + conv_tol) and all(
        prop_cum_mean[-last_m:] > mean_prop - conv_tol
    ):
        converged_cum_mean = True

    else:
        if verbose:
            print("The cumulative property mean does not converge to the global mean.")

    if converged_last_val and converged_cum_mean:
        return True

    else:
        return False


def determine_discard_number(
    property_array, init_discard=None, increment=10, verbose=False
):
    """Determine the minimum number of samples to discard for MC equilibration.

    Identify a range of number of samples to discard. Determine the smallest value that
    yields a converged property.

    Args:
        property_array (ndarray):
            array of numerical values of a property (energy, composition, etc)
        init_discard (int):
            initial guess for number of samples to discard. If None, take the first 10%
            as the initial guess
        increment (int):
            determines the increment of discard values to search for. Will search for
            discard values in increments of (n_samples - init_discard)/increment

    Returns:
        discard_n (int):
            number of samples to discard

    """
    property_array = np.array(property_array)
    n_samples = len(property_array)

    if init_discard is None:
        init_discard = int(n_samples / 10)
    elif init_discard > n_samples:
        warn(
            f"Number to discard ({init_discard}) is greater than num samples "
            f"({n_samples})! Will set the discard value to be 10% of the number of"
            f"samples ({int(n_samples/10)})"
        )
        init_discard = int(n_samples / 10)

    discard_val_increment = (n_samples - init_discard) / increment
    for discard_n in np.arange(
        init_discard, n_samples - discard_val_increment, discard_val_increment
    ):
        discard_n = int(discard_n)
        if check_property_converged(property_array[discard_n:]):
            return discard_n

    if verbose:
        print(
            "Could not find a discard value that leads to MC convergence. Will return "
            "0."
        )

    return 0
