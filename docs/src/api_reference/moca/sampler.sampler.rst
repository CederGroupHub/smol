.. _sampler:

=======
Sampler
=======

The :class:`Sampler` is the main class users will interact with
for MC sampling and is the holder of all important classes
needed to perform MC sampling. In general, it is recommended
to use the :meth:`from_ensemble` function to automatically
instantiate the relevant classes (:class:`MCKernel`,
:class:`MCUsher`, :class:`SamplerContainer`) to perform
Metropolis sampling and either single site flipping or
two-site swapping depending on the ensemble.

.. autoclass:: smol.moca.sampler.sampler.Sampler
   :members:
   :undoc-members:
   :show-inheritance:
