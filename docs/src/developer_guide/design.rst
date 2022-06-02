.. _design :

==============
Package Design
==============

Overview & mission
==================

**smol** is intentionally designed to be easy to use, install and extend. In order to
achieve these goals the package has few dependencies [#f1]_, and has a heavily
object-oriented and modular design that closely follows mathematical and methodological
abstractions. This enables flexible creation of complex workflows and hassle-free
implementation methodology extensions, that will rarely need to be implemented from
scratch.

**smol** has been designed to enable efficient and open development of new methodology
for fitting and sampling applied lattice models in a user-friendly way; and thus
allow quick development-to-application turnaround time in the study of configuration
dependent properties of inorganic materials.

Module Design
=============

smol.cofe
---------

smol.moca
---------

.. rubric:: Footnotes

.. [#f1] The dependence on **pymatgen** implicitly includes all of its dependecies---
         which are many. However, once **pymatgen** is properly installed, then
         installing **smol** should be headache free.
