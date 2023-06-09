.. _citing :

======
Citing
======

.. note::
    If you have any comments, suggestions or you think we have missed a
    pertinent publication in our recommended citations, please let us know!


If you use **smol** for your work please cite the following publication,

    Barroso-Luque, L., Yang, J.H., Xie, F., Chen T., Kam, R.L., Jadidi, Z., Zhong, P.
    & Ceder, G.
    `smol: A Python package for cluster expansions and beyond
    <https://joss.theoj.org/papers/10.21105/joss.04504>`_.
    Journal of Open Source Software 7, 4504 (2022).

Since many core objects are used extensively, please also consider citing the
**pymatgen** publication,

    Ong, S. P. et al. Python Materials Genomics (pymatgen):
    `A robust, open-source python library for materials analysis
    <https://doi.org/10.1016/j.commatsci.2012.10.028>`_.
    ComputationalMaterials Science 68, 314–319 (2013).

Generally, we recommend that any work using the cluster expansion method cite the
original publication,

    Sanchez, J. M., Ducastelle F., & Gratias D.
    `Generalized cluster description of multicomponent systems
    <https://doi.org/10.1016/0378-4371(84)90096-7>`_.
    Physica A: Statistical Mechanics and its Applications 128, 334–350 (1984).

Cluster expansion site basis sets
=================================

Based on the specific site basis sets used in the cluster expansion, we recommend citing
the following papers.

When using :code:`sinusoid` site basis sets consider citing,

    van de Walle, A. `Multicomponent multisublattice alloys, nonconfigurational entropy
    and other additions to the Alloy Theoretic Automated Toolkit.
    <https://doi.org/10.1016/j.calphad.2008.12.005>`_ Calphad 33, 266–278 (2009).

When using :code:`indicator` site basis sets consider citing,

    Zhang, X. & Sluiter, M. H. F. `Cluster Expansions for Thermodynamics and Kinetics of
    Multicomponent Alloys <https://doi.org/10.1007/s11669-015-0427-x>`_.
    J. Phase Equilib. Diffus. 37, 44–52 (2016).


Cluster decomposition
=====================

If you use the :class:`ClusterDecompositionProcessor` class, cluster interaction tensors,
cluster weights, or related concepts in your work, please cite this work,

    Barroso-Luque, Luis & Ceder, G. (2023).
    The cluster decomposition of the configurational energy of multicomponent alloys.
    `[arXiv] <https://doi.org/10.48550/arXiv.2301.02309>`_


Cluster expansions with redundant function sets
===============================================

If you use the :class:`PottsSubspace` class or related functionality
to include redundancy in cluster expansions, please cite this publication,

    Barroso-Luque, L., Yang, J. H. & Ceder, G.
    `Sparse expansions of multicomponent oxide configuration energy using
    coherency and redundancy
    <https://link.aps.org/doi/10.1103/PhysRevB.104.224203>`_.
    Phys. Rev. B 104, 224203 (2021).


Special quasi-random structures (SQS)
=====================================

When generating special quasi-random structures (SQS), please cite the original
publication,

    Zunger, A., Wei, S.-H., Ferreira, L. G. & Bernard, J. E.
    `Special quasirandom structures. <https://doi.org/10.1103/PhysRevLett.65.353>`_
    Phys. Rev. Lett. 65, 353–356 (1990).

If you use the :class:`StochasticSQSGenerator` class, please cite the following
publication,

    van de Walle, A. et al.
    `Efficient stochastic generation of special quasirandom structures.
    <https://doi.org/10.1016/j.calphad.2013.06.006>`_
    Calphad 42, 13–18 (2013).


Coulomb electrostatic interactions
==================================
When using an :class:`EwaldTerm` as an additional term in a lattice Hamiltonian, please
cite the following publications,

    Seko, A. & Tanaka, I. `Cluster expansion of multicomponent ionic systems with
    controlled accuracy: importance of long-range interactions in heterovalent ionic
    systems <https://doi.org/10.1088/0953-8984/26/11/115403>`_.
    J. Phys.: Condens. Matter 26, 115403 (2014).

    Richards, W. D., Wang, Y., Miara, L. J., Kim, J. C. & Ceder, G.
    `Design of Li1+2xZn1−xPS4, a new lithium ion conductor
    <https://doi.org/10.1039/C6EE02094A>`_. Energy Environ. Sci. 9, 3272–3278 (2016).

    Barroso-Luque, L. et al.
    `Cluster expansions of multicomponent ionic materials: Formalism and methodology.
    <https://doi.org/10.1103/PhysRevB.106.144202>`_.
    Phys. Rev. B 106, 144202 (2022).


Charge-neutral semigrand canonical sampling
===========================================

If you use the charge neutral semigrand canonical sampling or any related functionality
please cite the following work,

    Xie, F., Zhong, P., Barroso-Luque, L., Ouyang, B. & Ceder, G.
    `Semigrand-canonical Monte-Carlo simulation methods for charge-decorated cluster
    expansions <https://doi.org/10.1016/j.commatsci.2022.112000>`_
    Computational Materials Science 218, 112000 (2023).


Wang-Landau sampling
====================

If you use Wang-Landau sampling to estimate density of states and derivative properties
we recommend citing the original publications,

    Wang, F. & Landau, D. P.
    `Efficient, Multiple-Range Random Walk Algorithm to Calculate the Density of States.
    <https://doi.org/10.1103/PhysRevLett.86.2050>`_.
    Phys. Rev. Lett. 86, 2050–2053 (2001).


Orbit group-wise regularized fits
=================================

If you use **smol** or any of the other great CE packages to fit an expansion
using orbit group wise regularization with Group Lasso or any related regression
model, please cite the following,

    Yang, J. H., Chen, T., Barroso-Luque, L., Jadidi, Z. & Ceder, G.
    `Approaches for handling high-dimensional cluster expansions of ionic systems
    <https://www.nature.com/articles/s41524-022-00818-3>`_.
    npj Comput Mater 8, 1–11 (2022).

    Barroso-Luque, L. et al.
    `Cluster expansions of multicomponent ionic materials: Formalism and methodology.
    <https://doi.org/10.1103/PhysRevB.106.144202>`_.
    Phys. Rev. B 106, 144202 (2022).


:math:`\ell_0\ell_2` MIQP fits with hierarchical constraints
============================================================

Similarly, if you use :math:`\ell_0\ell_2` mixed integer quadratic programming
for hierarchical constrained expansion fits, please cite these publications,

    Zhong, P., Chen, T., Barroso-Luque, L., Xie, F. & Ceder, G.
    An :math:`\ell_0\ell_2`-norm `regularized regression model for construction of
    robust cluster expansions in multicomponent systems
    <https://doi.org/10.1103/PhysRevB.106.024203>`_
    Phys. Rev. B 106, 024203 (2022).

    Barroso-Luque, L. et al.
    `Cluster expansions of multicomponent ionic materials: Formalism and methodology.
    <https://doi.org/10.1103/PhysRevB.106.144202>`_.
    Phys. Rev. B 106, 144202 (2022).
