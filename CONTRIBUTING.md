# Contributing & Developing

The best code comes from organized collaboration. Collaborations can be grossly
broken up into the categories below. Both are equally as important to create
and improve software. Please consider contributing in any way you can!

## Bugs, issues, input, questions, etc
Please use the 
[issue tracker](https://github.com/CederGroupHub/smol/issues) to share any
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

## Developing
Code contributions can be anything from fixing the simplest bugs, to adding new
extensive features or subpackages. If you have written code or want to start
writing new code that you think will improve `smol` then please follow the
steps bellow to add it to the repository. 

1.  `Fork` the repository and then `clone` your fork to your local workspace.

2.  In order to have changes available without having to re-install the package,
type in the top level directory:

        python setup.py develop

3.  If at any point you need to re-cythonize (in case you changed a .pyx file
while developing or the .c files have gone missing) use:

        python setup.py develop --use-cython

4.  Once you have changes worth sharing open a `pull request` to the `develop`
branch. The `master` branch will be reserved only for the most stable version
and will only be updated with the `develop` branch once the team is comfortable
with the stability of the code.
    -   When openning a new pull request you will get a standard template, you do
    not need to fill/keep everything, only the things that apply to your pull
    request, but please try to be as thorough as possible in describing what 
    you have done.
    -   You should make sure your work does not break any of the unittests, and
    if you implemented new functionality you should write tests for it.
        -   To run the tests you need to install `pytest`. Then from the top
         level directiory run:
        
                pytest tests

        -   New tests should be put in the `/tests/` directory, and follow standard
        Python `unittest` naming.
    -   You are also encouraged to start a pull request before you're fully done
    with your work when you want others to see what you are doing and possibly
    get feedback or comments. If you open a pull request that is a work in
    progress use [WIP] in the title.
    -   If you are working on a large feature then you should consider creating a
    new branch so that your work can be included in the main repo during
    development (once the feature is ready in can be merged into the `develop`
    branch). To open up a new branch contact a repo admin.
