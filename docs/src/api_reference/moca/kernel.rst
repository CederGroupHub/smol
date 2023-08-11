=======
Kernels
=======

An instance of :class:`MCKernel` specifies the Monte Carlo
algorithm to be used for sampling. The :class:`MCKernel` class
use :class:`MCUsher` to generate step proposals for MC sampling. A
:class:`MCBias` can also be added to bias the system towards,
for example, a given fugacity. Users will rarely need to instantiate
or use any of these classes directly, but instead create them
implicitly by using instantiating a :class:`Sampler`.

.. toctree::
   :maxdepth: 2

   kernel.kernels
   kernel.mcusher
   kernel.bias
