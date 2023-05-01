.. _ensemble:

========
Ensemble
========

This module contains the :class:`Ensemble` class for the different types of ensembles
that can be implemented for MC sampling. Systems at constant
temperature and constant composition (no species flow in/out of
the system) should have ``chemical_potentials`` set to ``None`` and choose an
appropriate step type (i.e. a swap).
Systems at constant temperature and constant applied chemical potential
(species flow allowed) should simple set the ``chemical_potentials``
property.

.. automodule:: smol.moca.ensemble
   :members: Ensemble
   :undoc-members:
   :show-inheritance:
