# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
from datetime import date

from smol import __version__

sys.path.insert(0, os.path.abspath("../../smol"))
sys.path.insert(0, os.path.abspath("notebooks"))

# -- Project information -----------------------------------------------------

project = "smol"
copyright = f"2022-{date.today().year}, Ceder Group"
author = "Luis Barroso-Luque"

# The full version, including alpha/beta/rc tags
release = __version__
version = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    # "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    # "sphinx.ext.linkcode",
    "nbsphinx",
    "sphinx_copybutton",
    "IPython.sphinxext.ipython_console_highlighting",
]

# Generate the API documentation when building
autosummary_generate = True
add_module_names = False
autoclass_content = "both"

napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_custom_sections = None

# Add any paths that contain templates here, relative to this directory.
source_suffix = [".rst"]

# The encoding of src files.
source_encoding = "utf-8"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to src directory, that match files and
# directories to ignore when looking for src files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "build",
    "Thumbs.db",
    ".DS_Store",
    "notebooks/.ipynb_checkpoints",
]
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "pydata_sphinx_theme"


html_theme_options = {
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/CederGroupHub/smol",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_align": "content",  # [left, content, right] For testing that the navbar
    "navbar_start": ["navbar-logo", "navbar-version"],
    "show_toc_level": 2,
    "navigation_depth": 2,
    "show_nav_level": 2,
    "use_edit_page_button": True,
    "primary_sidebar_end": ["indices.html", "sidebar-ethical-ads.html"],
    "external_links": [
        {
            "name": "Changes",
            "url": "https://github.com/CederGroupHub/smol/releases",
        },
        {"name": "Issues", "url": "https://github.com/CederGroupHub/smol/issues"},
    ],
    "content_footer_items": ["last-updated"],
}

html_context = {
    "github_user": "CederGroupHub",
    "github_repo": "smol",
    "github_version": "main",
    "doc_path": "docs/src",
    "source_suffix": source_suffix,
    "default_mode": "auto",
    "versions_dropdown": {
        "latest": "devel (latest)",
        "stable": "current (stable)",
    },
}

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
# html_style = ''

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ["_static"]

html_js_files = [
    "require.js",  # Add to your _static
    "custom.js",
]

html_css_files = [
    "css/smol.css",
]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Content template for the index page.
html_index = "index.html"

# If false, no module index is generated.
html_use_modindex = True

html_file_suffix = ".html"

# If true, the reST sources are included in the HTML build as _sources/<name>.
html_copy_source = False

# Output file base name for HTML help builder.
htmlhelp_basename = "smol"

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = 'docs/src/' + env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="https://github.com/CederGroupHub/smol/tree/v{{ env.config.release|e }}/{{ docname|e }}">{{ docname|e }}</a>.<br>
      <script>
        if (document.location.host) {
          $(document.currentScript).replaceWith(
            '<a class="reference external" ' +
            'href="https://nbviewer.jupyter.org/url' +
            (window.location.protocol == 'https:' ? 's/' : '/') +
            window.location.host +
            window.location.pathname.slice(0, -4) +
            'ipynb">View & download notebook in <em>nbviewer</em></a>.'
          );
        }
      </script>
    </div>

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""
