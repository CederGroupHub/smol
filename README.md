# SMoL
## Statistical Mechanics on Lattices
![CircleCI](https://img.shields.io/circleci/build/gh/CederGroupHub/smol/master?logo=circleci&style=for-the-badge&token=96d0d7a959e1e12044ff45daa43218ae7fa4303e)

Lighthweight but caffeinated implementations of computational methods for statistical mechanical calculations of configurational states for crystalline material systems.

### Installing
As any python package from source, go to the top level directory and type:

    pip install .

Or if you want to have changes available without re-installing:

    python setup.py develop

If you need to re-cythonize (in case you changed a .pyx file while developing or the .c files have gone missing)

    python setup.py develop --use-cython

### Contributing
Please consider contributing if you have even the slightest idea. The very best software comes from collaborative efforts.
* In case you have an idea, recommendation, problem or found a bug, but nothing actually down on code, use the [issue tracker](https://github.com/CederGroupHub/smol/issues).
* If you have lovely bits of code you want to contribute then please fork + pull request [github-flow](https://guides.github.com/introduction/flow/) or [git-flow](https://nvie.com/posts/a-successful-git-branching-model/) style.

#### Credits
Feel free to add yourself here for your contributions :)
* Will Richards - A good amount of the Cluster Expansion code is based on Will's cluster expansion module in the pyabinitio repository.
* Daniil Kitcheav - The LNO example data was also computed by Daniil.
