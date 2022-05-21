.. _cluster space:

==============
Cluster Spaces
==============

Implementation of ClusterSubspace and related PottsSubspace classes.

:class:`ClusterSubspace` is the workhorse for generating the objects and
information necessary for a cluster expansion. It contains the finite set
of orbits and orbit basis functions to be included in the cluster expansion.
The :class:`PottsSubspace` is an (experimental) class that is similar, but
diverges from the CE mathematic formalism.

ClusterSubspace
---------------

.. autoclass:: smol.cofe.space.clusterspace.ClusterSubspace
   :members:
   :undoc-members:
   :show-inheritance:

PottsSubspace
-------------

A :class:`PottsSubspace` implements the following expansion:

.. math::
   H(\sigma) = \sum_{\alpha\in D[N]} m_{\alpha}\mathbf{1}_{\alpha}(\sigma)

.. autoclass:: smol.cofe.space.clusterspace.PottsSubspace
   :members:
   :undoc-members:
   :show-inheritance:
