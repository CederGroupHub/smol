.. _processors:

==========
Processors
==========

This module contains classes for the processors used to
efficiently calculate energies of occupancies of a given
:class:`ClusterSubspace` in a given supercell. For cluster
expansions without an external (Ewald) term,
:class:`ClusterExpansionProcessor` is used. For those with an Ewald term,
a :class:`ClusterExpansionProcessor` is combined with a
:class:`EwaldProcessor` into a :class:`CompositeProcessor`.
This class should generally be instantiated with
:meth:`from_cluster_expansion`, which will automatically
identify what type of processor is needed.

.. toctree::
   :maxdepth: 2

   processors.base
   processors.composite
   processors.ewald
   processors.expansion
