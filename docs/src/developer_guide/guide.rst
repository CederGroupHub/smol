====================================
Contributing & Developing Guidelines
====================================

The best code comes from organized collaboration. Collaborations can be grossly
broken up into the categories below. Both are equally as important to create
and improve software. Please consider contributing in any way you can!

Bugs, issues, input, questions, etc
===================================
Please use the
`issue tracker <https://github.com/CederGroupHub/smol/issues>`_ to share any
of the following:

-   Bugs
-   Issues
-   Questions
-   Feature requests
-   Ideas
-   Input

Having these reported and saved in the issue tracker is very helpful to make
sure that they are properly addressed. Please make sure to be as descriptive
and neat as possible when opening up an issue, but remember any contribution is
much better than nothing!

Developing
==========
Code contributions can be anything from fixing the simplest bugs, to adding new
extensive features or subpackages. If you have written code or want to start
writing new code that you think will improve **smol** then please follow the
steps below to make a contribution. To know where best to start implementing new
functionality please have a look at the :ref:`design` page.

Guidelines
----------

* All code should have unit tests.
* Code should be well documented following `google style <https://google.github.io/styleguide/pyguide.html>`_  docstrings.
* All code should pass the pre-commit hook. The code follows the `black code style <https://black.readthedocs.io/en/stable/>`_.
* Additional dependencies should only be added when they are critical or if they are
  already a :mod:`pymatgen` dependency. More often than not it is best to avoid adding
  a new dependency by simply delegating to directly using the external packages rather
  than adding them to the source code.
* Implementing new features should be more fun than tedious.

Installing a development version
--------------------------------

#. *Clone* the main repository or *fork* it and *clone* clone your fork using git.
   If you plan to contribute back to the project, then you should create a fork and
   clone that::

        git clone https://github.com/<USER>/smol.git

   Where ``<USER>`` is your github username, or if you are cloning the main repository
   directly then ``<USER> = CederGroupHub``.

#. Install Python 3.8 or higher. We recommend using
   `conda <https://docs.conda.io/en/latest/>`_.

#. We recommend developing using a virtual environment. You can do so using
   `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
   or using `virtualenv <https://docs.python.org/3/tutorial/venv.html>`_.

#. Install a C compiler with `OpenMP <https://en.wikipedia.org/wiki/OpenMP>`_ support.
   You can find helpful instructions to do so for
   `Linux <https://scikit-learn.org/dev/developers/advanced_installation.html#compiler-linux>`_,
   `MacOS <https://scikit-learn.org/dev/developers/advanced_installation.html#compiler-macos>`_,
   `Windows <https://scikit-learn.org/dev/developers/advanced_installation.html#compiler-windows>`_,
   and `FreeBSD <https://scikit-learn.org/dev/developers/advanced_installation.html#compiler-freebsd>`_.
   system in the
   `scikit-learn's <https://scikit-learn.org/dev/index.html>`_ package documentation.
   You can still install *smol* without OpenMP support for development, but you will
   not benefit from parallelization when computing correlations and cluster
   interactions.

#. Install the development version of *smol* in *editable* mode::

    pip install --verbose --editable .[dev,test]

   This will install the package in *editable* mode, meaning that any changes
   you make to the source code will be reflected in the installed package.

Adding code contributions
-------------------------

#.  If you are contributing for the first time:

    * Install a development version of *smol* in *editable* mode as described above.
    * Make sure to also add the *upstream* repository as a remote::

        git remote add upstream https://github.com/CederGroupHub/smol.git

    * You should always keep your ``main`` branch or any feature branch up to date
      with the upstream repository ``main`` branch. Be good about doing *fast forward*
      merges of the upstream ``main`` into your fork branches while developing.

#.  In order to have changes available without having to re-install the package:

    * Install the package in *editable* mode::

         pip install -e .

    * If you make changes to cython files (a .pyx file) or the .c files have gone missing),
      the you will need to *re-cythonize*::

        python setup.py develop --use-cython


#.  To develop your contributions you are free to do so in your *main* branch or any feature
    branch in your fork.

    * We recommend to only your forks *main* branch for short/easy fixes and additions.
    * For more complex features, try to use a feature branch with a descriptive name.
    * For very complex feautres feel free to open up a PR even before your contribution is finished with
      [WIP] in its name, and optionally mark it as a *draft*.

#.  While developing we recommend you use the pre-commit hook that is setup to ensure that your
    code will satisfy all lint, documentation and black requirements. To do so install pre-commit, and run
    in your clones top directory::

        pre-commit install

    *  All code should use `google style <https://google.github.io/styleguide/pyguide.html>`_ docstrings
       and `black <https://black.readthedocs.io/en/stable/?badge=stable>`_ style formatting.

#.  Make sure to test your contribution and write unit tests for any new features. All tests should go in the
    ``smol\tests`` directory. The CI will run tests upon opening a PR, but running them locally will help find
    problems before::

        pytests tests


#.  To submit a contribution open a *pull request* to the upstream repository. If your contribution changes
    the API (adds new features, edits or removes existing features). Please add a description to the
    `change log <https://github.com/CederGroupHub/smol/blob/main/CHANGES.md>`_.

#.  If your contribution includes novel published (or to be published) methodology, you should also edit the
    :ref:`citing` page accordingly.


Adding examples
---------------

In many occasions novel use of the package does not necessarily require introducing new source code, but rather
using the existing functionality, and possibly external packages (that are are requirements) for particular or
advanced calculations.

#.  Create a notebook with a descriptive name in the ``docs/src/notebooks`` directory.
#.  Implement the functionality with enough markdown cells carefully describing the background, theory,
    and steps in the notebook.
#.  Any necessary data should be added to the ``docs/src/notebooks/data`` directory. Files should be at most
    a few MB.
#.  Once the notebook is ready, add an entry to the :ref:`getting-started` page so your notebook shows up in the
    documentation.
