==============
External Terms
==============

Additional pair interaction terms can be used to improve training
convergence. :class:`EwaldTerm` can be used to add an electrostatic
pair term based on the `Ewald summation
<https://pymatgen.org/pymatgen.analysis.ewald.html>`_, as
implemented in :mod:`pymatgen`, and is particularly useful in ionic systems such as
oxides.

.. _ewald term:

Ewald Term
----------

The Ewald term can be used to add an Ewald electrostatic pair term
to the fit. The Ewald sum is the sum of a reciprocal term, a real
term, and a point term. It is possible to fit to the full sum or
any of the individual terms.

.. automodule:: smol.cofe.extern.ewald
   :members:
   :undoc-members:
   :show-inheritance:
