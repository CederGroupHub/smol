=========
Expansion
=========

:class:`ClusterExpansion` contains the fitted coefficients of the cluster
expansion for predicting CE properties of new structures.

We provide a thin :class:`RegressionData` dataclass to record the specifics of
the regression used while fitting, for example when using a linear model from
`scikit-learn <https://scikit-learn.org/stable/modules/linear_model.html#linear-model>`_.

.. _cluster expansion:

ClusterExpansion
----------------

This module implements the ClusterExpansion class.

A ClusterExpansion holds the necessary attributes to represent a CE and predict
the property for new structures.

The class also allows pruning of the CE to remove low importance orbits function
terms and speed up Monte Carlo runs.

Also has numerical ECI conversion to other basis sets, but this functionality
has not been strongly tested.

.. autoclass:: smol.cofe.expansion.ClusterExpansion
   :members:
   :undoc-members:
   :show-inheritance:

RegressionData
--------------

.. autoclass:: smol.cofe.expansion.RegressionData
   :members:
   :undoc-members:
   :show-inheritance:
