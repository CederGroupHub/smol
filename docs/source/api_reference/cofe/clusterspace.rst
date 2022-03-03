==============
Cluster Spaces
==============

Some description of the module here. See the source code package docstring
in smol.cofe.space.clusterspace for what to put on here.

ClusterSubspace
---------------
Instead of using autodoc we can do individual class documentation as this:
And the we can also add some descriptions here about the specific classes.

.. autoclass:: smol.cofe.ClusterSubspace
   :members:
   :undoc-members:
   :show-inheritance:

PottsSubspace
-------------

Now the :class:`PottsSubspace`. We can write nice math too, with latex notation,
Inline as this :math:`H(\sigma) = \sum m_\beta\Theta_\beta(\sigma)`

A :class:`PottsSubspace` implements the following expansion:

.. math::
   H(\sigma) = \sum_{\alpha\in D[N]} m_{\alpha}\mathbf{1}_{\alpha}(\sigma)

.. autoclass:: smol.cofe.PottsSubspace
   :members:
   :undoc-members:
   :show-inheritance:

