========
Sampling
========

This module contains a number of classes for MC sampling. The
main modules users will generally interact with are
:class:`Sampler` and :class:`SampleContainer`. :class:`Sampler`
contains a :class:`MCKernel` to implement a specific MC
algorithm to sample the :class:`Ensemble` class as well as an
:class:`MCUsher` to generate step proposals for MC sampling. A
:class:`MCBias` can also be added to bias the system towards,
for example, a given fugacity. :class:`SampleContainer` contains
relevant properties of the samples observed during MC sampling,
such as occupancy, feature vector, energy, enthalpy, etc.

.. toctree::
   :maxdepth: 2

   sampler.sampler
   sampler.kernel
   sampler.mcusher
   sampler.container
   sampler.bias
