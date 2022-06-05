.. _citing :

======
Citing
======

If you use **smol** for your work please cite the following publication,

    Barroso-Luque, L., Yang, J.H., Xie, F., Chen T., Kam, R.L., Jadidi, Z., Zhong, P.
    & Ceder, G.
    `smol: A Python package for cluster expansions and beyond. (submitted)
    <https://joss.theoj.org/papers/e96a568ca53ee9d14548d7b8bed69b25>`_

Since many core objects are used extensively, please also consider citing the
**pymatgen** publication,

    Ong, S. P. et al. Python Materials Genomics (pymatgen):
    `A robust, open-source python library for materials analysis
    <https://doi.org/10.1016/j.commatsci.2012.10.028>`_.
    ComputationalMaterials Science 68, 314–319 (2013).

Cluster expansions with redundant function sets
===============================================

If you use the (:class:`smol.cofe.PottsSubspace`) class or related functionality
to include redundancy in cluster expansions please cite this publication,

    Barroso-Luque, L., Yang, J. H. & Ceder, G.
    `Sparse expansions of multicomponent oxide configuration energy using
    coherency and redundancy
    <https://link.aps.org/doi/10.1103/PhysRevB.104.224203>`_.
    Phys. Rev. B 104, 224203 (2021).

Orbit group wise regularized fits
=================================

If you use **smol** or any of the other great CE packages to fit an expansion
using orbit group wise regularization with Group Lasso or any related regression
model, please cite the following,

    Yang, J.H., Chen, T., Barroso-Luque, L., Jadidi, Z. & Ceder, G.
    Approaches for handling high-dimensional cluster expansions of ionic
    systems. (submitted)

    Barroso-Luque, L., Zhong, P., Yang, J. H., Chen, T. & Ceder, G.
    Cluster Expansions of Multicomponent Ionic Materials:Formalism & Methods.
    (in preparation)

:math:`\ell_0\ell_2` MIQP fits with hierarchical constraints
============================================================

Similarly, if you use :math:`\ell_0\ell_2` mixed integer quadratic programming
for hierarchical constrained expansion fits, please cite this publication,

    Zhong, P., Chen, T., Barroso-Luque, L., Xie, F. & Ceder, G.
    An :math:`\ell_0\ell_2` norm regularized regression model for construction
    of robust cluster expansion in multicomponent systems.
    `[arxiv] <https://arxiv.org/abs/2204.13789>`_
