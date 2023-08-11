=========
MCKernels
=========

:class:`MCKernel` classes implement different Monte Carlo sampling algorithms. They are
used by :class:`Sampler` classes to perform the actual sampling. The following kernels
are currently implemented:

.. automodule:: smol.moca.kernel.metropolis
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: MetropolisAcceptMixin

.. automodule:: smol.moca.kernel.wanglandau
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: smol.moca.kernel.random
   :members:
   :undoc-members:
   :show-inheritance:
