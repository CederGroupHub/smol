.. _sample container:

================
Sample Container
================

A :class:`SampleContainer` holds data (occupancies, feature vectors, energies)
sampled during MC sampling. A :class:`SampleContainer` is generally
automatically created when a :class:`Sampler` is instantiated. In addition to
holding sample information, the :class:`SamplerContainer` also provides
some methods for simple analysis of the raw data, including means,
variances, and minima of energies and enthalpies, mean compositions,
minimum-energy occupancies, etc.

.. autoclass:: smol.moca.sampler.SampleContainer
   :members:
   :undoc-members:
   :show-inheritance:
