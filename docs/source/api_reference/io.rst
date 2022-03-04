=======================
io --- Input and Output
=======================

automodule directive does the full module for you.

To control what goes into autodoc here edit the docstrings in the actual source
code.

I'd suggest using autodoc for the classes/functions that are not that important
or is very obvious/simple what their use case is.

.. automodule:: smol.io
   :members:
   :undoc-members:
   :show-inheritance:
