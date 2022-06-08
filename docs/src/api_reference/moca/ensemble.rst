.. _ensemble:

========
Ensemble
========

This module contains the :class:`Ensemble` class for the different types of ensembles
that can be implemented for MC sampling. Systems at constant
temperature and constant composition (no species flow in/out of
the system) should have :prop:`chemical_potentials` set to ``None``.
Systems at constant temperature and constant applied chemical potential
(species flow allowed) should simple set the :prop:`chemical_potentials`
property.

.. automodule:: smol.moca.ensemble.ensemble
   :members:
   :undoc-members:
   :show-inheritance:
