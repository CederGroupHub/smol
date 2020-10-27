# <img src="docs/images/logo.png" width="350px" alt="smol">

### Statistical Mechanics on Lattices

![CircleCI](https://img.shields.io/circleci/build/gh/CederGroupHub/smol/master?logo=circleci&style=for-the-badge&token=96d0d7a959e1e12044ff45daa43218ae7fa4303e)
![Codacy Badge](https://img.shields.io/codacy/coverage/4b527a2fd9ad40f59195f1f8dc1ac542?style=for-the-badge)
![Codacy Badge](https://img.shields.io/codacy/grade/4b527a2fd9ad40f59195f1f8dc1ac542?style=for-the-badge)

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
**smol** would not be possible without initial groundwork, inspiration,
contributions, and suggestions from the following people/projects... 
-   Will Richards - A good amount of the Cluster Expansion code is based on Will's cluster expansion module in the pyabinitio repository.
-   Daniil Kitcheav - The pyabinitio benchark test data and the LNO example data was computed by Daniil.
