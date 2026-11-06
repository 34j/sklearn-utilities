# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from pathlib import Path
from typing import Any

from sphinx.application import Sphinx
from sphinx.ext import apidoc

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Sklearn Utilities"
copyright = "2023, 34j"
author = "34j"
release = "0.5.14"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]
napoleon_google_docstring = False

source_suffix = [".rst", ".md"]
templates_path = ["_templates"]
exclude_patterns: list[str] = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


# -- Automatically run sphinx-apidoc -----------------------------------------


def run_apidoc(_: Any) -> None:
    docs_path = Path(__file__).parent
    module_path = docs_path.parent / "src" / "sklearn_utilities"

    apidoc.main(
        [
            "--force",
            "--module-first",
            "-o",
            docs_path.as_posix(),
            module_path.as_posix(),
        ]
    )


def setup(app: Sphinx) -> None:
    app.connect("builder-inited", run_apidoc)
