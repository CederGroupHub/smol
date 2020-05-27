# SMoL
## Statistical Mechanics on Lattices
![CircleCI](https://img.shields.io/circleci/build/gh/CederGroupHub/smol/master?logo=circleci&style=for-the-badge&token=96d0d7a959e1e12044ff45daa43218ae7fa4303e)
![Codacy Badge](https://img.shields.io/codacy/coverage/4b527a2fd9ad40f59195f1f8dc1ac542?style=for-the-badge)
![Codacy Badge](https://img.shields.io/codacy/grade/f6180b5223f346d2ac9dcf9a4bcc62d9?style=for-the-badge)

Lighthweight but caffeinated Python implementations of computational methods
for statistical mechanical calculations of configurational states for
crystalline material systems.

### Installing & Running
1. `Clone` the repository.
2. Go to the top level directory of the cloned repo and type:

        pip install .
3. See the [example notebooks](https://github.com/CederGroupHub/smol/tree/master/examples)
to help you get started.

### Contributing & Developing
Please consider contributing even the smallest slightest idea.
The very best software comes from collaborative efforts.

In case you have an idea, recommendation, problem or found a bug, but nothing
actually down on code, use the [issue tracker](https://github.com/CederGroupHub/smol/issues).

If you have lovely bits of code you want to contribute then please fork + pull
request [github-flow](https://guides.github.com/introduction/flow/) or
[git-flow](https://nvie.com/posts/a-successful-git-branching-model/) style:

1. `Fork` the repository.
2. In order to have changes available without having to re-install the package,
type in the top level directory:

        python setup.py develop
3. If at any point you need to re-cythonize (in case you changed a .pyx file
while developing or the .c files have gone missing) use:

        python setup.py develop --use-cython

4. Once you have changes worth sharing open a `pull request` to the `develop`
branch. You should make sure your work does not break any of the unittests, and
if you implemented new functionality you should write tests for it.
You are also encouraged to start a pull request before you fully done with your
work, if you others to see what you are doing and possibly give feedback or
comments. If you open a pull request that is a work in progress use [WIP] in
the title.

#### Credits
Feel free to add yourself here for your contributions :)
*   Will Richards - A good amount of the Cluster Expansion code is based on Will's cluster expansion module in the pyabinitio repository.
*   Daniil Kitcheav - The pyabinitio benchark test data and the LNO example data was computed by Daniil.
