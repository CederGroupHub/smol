.. _processors:

==========
Processors
==========

This module contains classes for the processors used to
efficiently calculate energies of occupancies of a given
:class:`ClusterSubspace` in a given supercell.

For cluster expansions without an external (Ewald) term, a :class:`ClusterDecompositionProcessor`
:class:`ClusterExpansionProcessor` is used. The :class:`ClusterDecompositionProcessor`
will result in faster sampling since it scales only with the number of orbits of
symmetrically distinct clusters as opposed to the number of correlation functions.
However if you need correlation function values from samples in your analysis then
you will need to use a :class:`ClusterExpansionProcessor`.

For calculations with an Ewald term,
either a :class:`ClusterDecompositionProcessor` or a :class:`ClusterExpansionProcessor`
is combined with a :class:`EwaldProcessor` into a :class:`CompositeProcessor`.
This class should generally be instantiated with
:meth:`from_cluster_expansion`, which will automatically
identify what type of processor is needed.

.. toctree::
   :maxdepth: 2

   processors.expansion
   processors.ewald
   processors.composite
