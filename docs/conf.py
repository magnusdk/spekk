import os
import sys

# Add spekk to path
sys.path.insert(0, os.path.abspath(".."))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "spekk"
copyright = "2023, Magnus Dalen Kvalevåg"
author = "Magnus Dalen Kvalevåg"
release = "1.0.9"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Generates documentation from docstrings
    "sphinx.ext.napoleon",  # Adds support for NumPy and Google style docstrings
    "sphinx.ext.autosummary",  # Automatically generates documentation for modules
    "sphinx.ext.viewcode",
    "sphinx_copybutton",  # Adds copy-button to code-blocks
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
