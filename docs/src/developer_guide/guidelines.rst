Contributing & Developing Guidelines
====================================

The best code comes from organized collaboration. Collaborations can be grossly
broken up into the categories below. Both are equally as important to create
and improve software. Please consider contributing in any way you can!

Bugs, issues, input, questions, etc
-----------------------------------
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
----------
Code contributions can be anything from fixing the simplest bugs, to adding new
extensive features or subpackages. If you have written code or want to start
writing new code that you think will improve ``smol`` then please follow the
steps bellow to add it to the repository.

1.  ``Fork`` the repository and then ``clone`` your fork to your local workspace.
    You should always keep your ``main`` branch up to date with the main
    repository ``main`` branch. Be good about doing *fast forward* merges of
    the main ``main`` into your ``main`` branch while developing.

2.  In order to have changes available without having to re-install the package,
type in the top level directory:

        pip install -e .

3.  If at any point you need to re-cythonize (in case you changed a .pyx file
while developing or the .c files have gone missing) use:

        python setup.py develop --use-cython

4. Once you have changes worth sharing there are 3 main ways of adding your
work to the main repository for others to see. All these require you to open a
***pull request*** to a specific branch in the main repo. If you are not sure
were your work should go please ask one of the repo administrators.
   1. For short and concise additions (such as single new methods, bug fixes
   and small edits) open a PR to the ``main`` branch.
   2. For more involved work in developing complex functionality, consider
   opening a new *feature* branch for your work that will eventually be merged
   into the ``main`` branch once all your work is ready.
   3. For small, but very specific or niche features; such that they are very
   "experimental" but are not general enough to or easy to add robust testing
   for open a PR to the ``experimental`` branch.

5. Since the ``main`` branch is reserved only for the most stable version of
the code, any new features should come with new unit-tests. Every so often a
tag will be added to create versions of the code where it is most
stable/tested. When opening a PR to the ``main`` branch, you will need to do
the following:
    -   When opening a new pull request you will get a standard template, you
    do not need to fill/keep everything, only the things that apply to your
    pull request, but please try to be as thorough as possible in describing
    what you have done.
    -   You should make sure your work does not break any of the unittests, and
    if you implemented new functionality you should write tests for it.
        -   To run the tests you need to install ``pytest``. Then from the top
         level directiory run:

                pytest tests

        -   New tests should be put in the ``/tests/`` directory, and follow
        standard ``pytest`` naming.
    -   You should also check your code conforms with standard python style by
        installing ``flake8`` (use pip). Then from the top level directory,

                flake8 smol
    -   Make sure you write proper docstrings following ***Google***
    style. You can just look at other docstrings to see what this looks like,
    but it is also helpful to set up your IDE to help you with this (PyCharm
    is nice for this). You should install ``pydocstyle`` and run it from the top
    level directory to make sure everything looks real nice.

                pydocstyle smol
    -   Finally, you are also encouraged to start a pull request before you're
    fully done with your work when you want others to see what you are doing
    and possibly get feedback or comments. If you open a pull request that is a
    work in progress use [WIP] in the title.

6. If you will working on a large feature you should consider discussing the
idea with the team first. Then you should consider creating a new **feature**
branch so that your work can be included in the main repo during development
(once the feature is ready in can be merged into the ``main`` branch). This
will allow you to work on your own schedule, but also allow for suggestions
and collaboration with others. To open up a new branch contact a repo admin.
    -   You need to tell the admin creating the name for your feature branch.
    Make sure this is a descriptive name so it is obvious to others what the
    branch is for.
    -   You can (and should) open many PRs to the feature branch as you are
    developing the necessary code. These PRs do not need anything special you
    are free to organize them however you see fit.
    - As you work on your feature make sure you follow updates on the
    ``main`` branch as closely as possible by doing appropariate
    fast-forward merges.
    - If you are working on a feature which needs changes or updates of any
    existing code please discuss this with the author before doing changes
    and adding work on those changes. This makes mergine larger features
    a whole lot easier.
    - If your feature needs another package that is not currently a requirement
    make sure to run it by @lbluque before putting in too much work. In most
    cases additional dependencies will not be accepted.
    -   When you think the code for the the branch is ready and well tested
    with its corresponding unit tests, then it can be merged into the ``main``
    branch to become part of the main code. This requires a PR from the feature
    branch to the ``main`` branch. Ask a repo admin to help you with this if
    necessary.
    -   Once the feature branch has been merged to the ``main`` branch it will
    be deleted.

 ### A note on style
 Keeping the coding style consistently clean makes using the code and further
 development a whole lot easier. When writing new code please look at existing
 code and do your best to follow the current coding style.
 [PEP8](https://www.python.org/dev/peps/pep-0008/) everywhere please.
 Also look into the
 [google style guide](https://google.github.io/styleguide/pyguide.html)
 (although everything in there is not strictly done here, it is a good guiding
 document).
 Name your variables, functions and classes with a descriptive name; and use
 abbreviations sparingly and make it *very obvious* what the abbrevation
 means. All new public classes and functions must have detailed docstring
 following
 [google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
 (private objects should still have a docstring but get a bit more slack).
 Doing this allows to easily keep the autogenerated documentation clean and up
 to date.
 Making sure all of the above is done continuously during development keeps the
 code in good shape for developers and users alike.
