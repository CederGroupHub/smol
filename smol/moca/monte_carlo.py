from __future__ import division

from math import exp
from monty.json import MSONable
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from random import random

import logging
import numpy as np


class MonteCarloRunner:
    """
    Typical usage for temperature scanning:
    mc = MonteCarloRunner(ecis=ecis, initial_occu=occu, cluster_supercell=cluster_supercell, flip_function=get_flips)
    mc.run_mc(40000, start_t=20, end_t=400, n_samples=20000)
    mc.run_mc(120000, start_t=400, end_t=6000, n_samples=60000)
    mc.run_mc(40000, start_t=6000, end_t=300000, n_samples=20000)

    mc.run_mc(40000, start_t=300000, end_t=6000, n_samples=20000)
    mc.run_mc(120000, start_t=6000, end_t=400, n_samples=60000)
    mc.run_mc(40000, start_t=400, end_t=20, n_samples=20000)

    dumpfn(mc.get_mc_data(), 'mcdata.mson')
    """
    k = 8.6173303e-05

    def __init__(self, initial_occu, cluster_supercell,
                 ecis, flip_function, corr_inds=None):
        """
        Args:
            initial_occu: initial occupancy (from cluster_supercell)
            ecis: ecis for cluster expansion
            flip_function: function that chooses flips based on a given occupancy
                (see ClusterSupercell.delta_corr)
            corr_inds (np.array[int]): Optional indices of correlations to track
                during simulation.
        """
        self.ecis = ecis
        self.initial_occu = initial_occu
        self.occu = initial_occu.copy()
        self.flip_function = flip_function
        self.energies = []
        self.temperatures = []
        self.corr_inds = corr_inds
        self.correlations = []
        self.e = cluster_supercell.occu_energy(self.occu, ecis)
        logging.debug("starting energy: {}".format(self.e))
        self.min_e = np.inf
        self.min_occu = initial_occu
        self.cs = cluster_supercell

    def run_mc(self, n_iterations, start_t, end_t, n_samples):
        """
        Runs Monte Carlo iterations between two temperatures.
        Args:
            n_iterations: total number of flips
            start_t, end_t: starting and ending temperature. log interpolated
            ending temperature
            n_samples: number of samples to store from these steps.
                Total number from all calls to run_mc should be ~100,000
        """
        sample_frequency = n_iterations // n_samples
        log_t = np.log(start_t)
        delta_log_t = (np.log(end_t) - log_t) / n_iterations
        trailing_e = self.e
        trail_x = 1 / (2 * sample_frequency)

        if self.corr_inds:
            tracked_corr = self.cs.corr_from_occupancy(self.occu)[self.corr_inds]
            trailing_corr = tracked_corr.copy()

        for loop in xrange(n_iterations):
            flips = self.flip_function(self.occu)
            d_corr, new_occu = self.cs.delta_corr(flips, self.occu)
            de = np.dot(d_corr, self.ecis) * self.cs.size
            if de < 0 or random() < exp(-de / exp(log_t) / self.k):
                self.occu = new_occu
                self.e += de
                if self.corr_inds:
                    tracked_corr += d_corr[self.corr_inds]
            trailing_e = (1 - trail_x) * trailing_e + trail_x * self.e
            if self.corr_inds:
                trailing_corr = (1 - trail_x) * trailing_corr + trail_x * tracked_corr
            if loop % 10000 == 0:
                logging.debug("{}, energy: {}, moving_avg: {}, temperature: {}"
                              "".format(loop, self.e, trailing_e, exp(log_t)))
                # recalculate energy and tracked correlations
                self.e = self.cs.occu_energy(self.occu, self.ecis)
                tracked_corr = self.cs.corr_from_occupancy(self.occu)[self.corr_inds]
            if loop % sample_frequency == 0:
                self.energies.append(trailing_e)
                self.temperatures.append(exp(log_t))
                if self.corr_inds:
                    self.correlations.append(trailing_corr)
            if self.e < self.min_e:
                self.min_e = self.e
                self.min_occu = self.occu
            log_t += delta_log_t

    @property
    def min_e_structure(self):
        return self.cs.structure_from_occu(self.min_occu)

    @property
    def data(self):
        """
        Get all the monte carlo data as a numpy array of temperatures and energies
        """
        return np.array([self.temperatures, self.energies])

    def get_mc_data(self, n_samples=10000, initial_S=None):
        if initial_S is None:
            if self.temperatures[0] > 50:
                raise ValueError('Must supply an initial_S for runs that start at '
                                 'finite temperature. If starting from infinite T, '
                                 'this is the ideal mixing entropy.')
            logging.info('Calculating initial_S based on symmetry of '
                         'starting structure and supercell')
            initial_struct = self.cs.structure_from_occu(self.initial_occu)
            smin_nops = len(SpacegroupAnalyzer(initial_struct).get_symmetry_operations())
            supercell_nops = len(SpacegroupAnalyzer(self.cs.supercell).get_symmetry_operations())
            initial_S = self.k * np.log(supercell_nops / smin_nops)

        return MCData.from_raw_data(self.temperatures, self.energies,
                                    initial_S, n_samples=n_samples)


class MCData(MSONable):
    """
    Stores Monte Carlo data, and calculates thermo data from a temperature
    scanning run
    """
    def __init__(self, T, E, S, G):
        """
        Don't use this constructor usually. Only needed for as_dict and from_dict
        """
        self.T = np.array(T)
        self.E = np.array(E)
        self.S = np.array(S)
        self.G = np.array(G)

    @classmethod
    def from_raw_data(cls, temperatures, energies, initial_S, n_samples=10000):
        """
        Args:
            temperatures (np.array): temperature at each sampling step
            energies (np.array): internal energy at each sampling step
            initial_S (np.array): starting entropy
            n_samples (int): number of data points to store
        """
        e = np.array(energies)
        t = np.array(temperatures)
        dQ = e[1:] - e[:-1]
        dS = dQ / t[1:]
        S = np.cumsum(dS) + initial_S
        G = e[1:] - t[1:] * S

        # make them all the same length
        skip = max(len(G) // n_samples, 1)
        max_len = min(n_samples, len(e))
        return MCData(t[::skip][:max_len], e[::skip][:max_len],
                      S[::skip][:max_len], G[::skip][:max_len])
