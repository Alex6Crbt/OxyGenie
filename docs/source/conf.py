# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

# sys.path.insert(0, os.path.abspath('../simulation/diffusion_simulation'))
# sys.path.insert(0, os.path.abspath('../simulation/procedural_gen_vascular'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OxyGenie'
copyright = '2024, A. CORBILLET, I. TOUAIMIA'
author = 'A. CORBILLET, I. TOUAIMIA'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_favicon",
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_logo = "_static/logo.png"
html_context = {
    "default_mode": "light",
}

html_theme_options = {
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "icon_links": [
    {
        "name": "GitHub",
        "url": "https://github.com/Alex6Crbt/OxyGenie",
        "icon": "fa-brands fa-github",
    },],
}
