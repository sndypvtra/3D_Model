# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = 'Hunyuan3D-2'
copyright = '2025, Tencent Hunyuan3D'
author = 'Hunyuan3D Team'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'nbsphinx',
    'nbsphinx_link',
    # "myst_nb",
    'sphinx_copybutton',
    # "sphinx_inline_tabs",
    # https://sphinx-codeautolink.readthedocs.io/en/latest/examples.html
    'sphinx.ext.autodoc',
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

# -- Options for extlinks ----------------------------------------------------
#

extlinks = {
    "pypi": ("https://pypi.org/project/%s/", "%s"),
}

# -- Options for intersphinx -------------------------------------------------
#

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    'torch': ('https://pytorch.org/docs/master/', None)
}

napoleon_preprocess_types = True

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'sphinx_rtd_theme'
html_theme = "furo"
html_title = "Hunyuan3D-2"
language = "en"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    "light_css_variables": {
        "font-stack": "Arial,Noto Sans,sans-serif",
        "font-stack--monospace": "IBM Plex Mono,ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace",
    },
    "announcement": 'Release 🤗<a href="https://huggingface.co/spaces/tencent/Hunyuan3D-2mini-Turbo">Turbo Series</a> and <a href="https://github.com/Tencent/FlashVDM">FlashVDM</a>, Fast Shape Generation within 1 Second Right Now!',
}

#
# -- Options for TODOs -------------------------------------------------------
#
todo_include_todos = True

#
# -- Options for Markdown files ----------------------------------------------
#
myst_admonition_enable = True
myst_deflist_enable = True
myst_heading_anchors = 3

html_favicon = '_static/favicon.ico'

pygments_style = "default"
pygments_dark_style = "github-dark"

html_css_files = [
    'css/custom.css',
]
