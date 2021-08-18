# <img src="docs/images/logo.png" width="350px" alt="smol">

### Statistical Mechanics on Lattices

![test](https://github.com/CederGroupHub/smol/actions/workflows/test.yml/badge.svg)

Lighthweight but caffeinated Python implementations of computational methods
for statistical mechanical calculations of configurational states for
crystalline material systems.

**Documentation (requires lbl vpn): <http://amox.lbl.gov/smol>**

#### Installing & Running
1.  `Clone` the repository.
    -   The latest tag in the `master` branch is the stable version of the
    code. The `master` branch has the newest tested features, but may have more
    lingering bugs.

2.  Go to the top level directory of the cloned repo and type:

        pip install .

3.  See the [example notebooks](https://github.com/CederGroupHub/smol/tree/master/examples)
to help you get started.

#### Contributing & Developing
Please consider contributing even the smallest idea.
The very best software comes from collaborative efforts.

In case you have an idea, recommendation, problem or found a bug, but nothing
actually down on code, use the [issue tracker](https://github.com/CederGroupHub/smol/issues).

If you have lovely bits of code you want to contribute then please fork + pull
request [github-flow](https://guides.github.com/introduction/flow/) or
[git-flow](https://nvie.com/posts/a-successful-git-branching-model/) style.
See the contributing
[file](https://github.com/CederGroupHub/smol/blob/master/CONTRIBUTING.md) for
further information.

#### Credits
**smol** would not be possible without initial groundwork, contributions, and
inspiration, from the following people/projects:
-   The Ceder Group Cluster Expansion team and all the contributors listed in
    this repository have provided invaluable contributions, suggestions, ideas,
    fixes and feedback that have continuosly improved **smol**.
-   Will Richards & Daniil Kitcheav - Provided the initial code and inspiration
    based their cluster expansion module in the pyabinitio repository.
-   [pymatgen](https://pymatgen.org/) makes up the backbone of **smol**, by
    providing the objects to represent compositions, structures, and symmetry
    operations.
-   A fair deal of the design of the **smol.cofe** module is inspired by
    [icet](https://icet.materialsmodeling.org/) another amazing and highly
    recommended CE python package, that has many great features that are not
    supported here.
-   A handful of ideas for the design of **smol.moca** were borrowed from many
    great quality probabilistic programming packages such as
    [mcpele](http://pele-python.github.io/mcpele/),
    [TensorFlow Probability](https://www.tensorflow.org/probability),
    [Pyro](https://pyro.ai/), [pyABC](https://pyabc.readthedocs.io/en/latest/),
    and [emcee](https://emcee.readthedocs.io/en/stable/).
    

