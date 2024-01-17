# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os

project = "OrdinalGBT"
copyright = "2023, Adamos Spanashis"
author = "Adamos Spanashis"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
]
if os.environ.get("NO_MATHJAX",False):
    extensions.append( "sphinx.ext.imgmath")
    imgmath_latex_preamble = "\\usepackage{amsmath}"
else:
    extensions.append( "sphinx.ext.mathjax")
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
autoapi_dirs = ["../ordinalgbt"]  # location to parse for API reference
html_theme = "sphinx_rtd_theme"
exclude_patterns = []
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

nb_execution_mode = "cache"
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
