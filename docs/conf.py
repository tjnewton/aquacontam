"""Sphinx configuration for AquaContam API docs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

project = "AquaContam"
author = "Tyler J. Newton"
release = "4.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
]

# Napoleon settings
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Theme
html_theme = "sphinx_rtd_theme"

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
