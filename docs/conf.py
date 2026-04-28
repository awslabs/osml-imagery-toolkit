# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "osml-imagery-toolkit"
copyright = "Amazon.com, Inc."
author = "AWS OSML"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.mermaid",
]

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "fieldlist",
]
myst_fence_as_directive = ["mermaid"]

# Autodoc settings
autodoc_member_order = "bysource"

# Intersphinx for linking to NumPy, Shapely docs, etc.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Theme
html_theme = "furo"

# GitHub Pages base URL
html_baseurl = "https://awslabs.github.io/osml-imagery-toolkit/"

# Static files (images, custom CSS, etc.)
html_static_path = ["_static"]

# Exclude internal working notes from the published site
exclude_patterns = ["_build", "internal", "Thumbs.db", ".DS_Store"]

# Suppress warnings from autodoc for optional dependencies and decorated method signatures
suppress_warnings = ["autodoc.import_error", "autodoc"]

# -- LaTeX / PDF output configuration ----------------------------------------

latex_engine = "pdflatex"

latex_documents = [
    (
        "index",  # startdocname
        "osml-imagery-toolkit.tex",  # targetname
        "osml-imagery-toolkit Documentation",  # title
        "AWS OSML",  # author
        "manual",  # theme ('manual' or 'howto')
    ),
    (
        "user-guide/index",  # startdocname
        "osml-imagery-toolkit-user-guide.tex",  # targetname
        "osml-imagery-toolkit User Guide",  # title
        "AWS OSML",  # author
        "manual",  # theme
    ),
]

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "11pt",
    # Remove blank pages between chapters for a more compact PDF
    "extraclassoptions": "openany,oneside",
    # Custom preamble: handle Unicode chars that pdflatex can't render natively
    "preamble": r"""
\usepackage{enumitem}
\setlistdepth{99}
\usepackage{newunicodechar}
\newunicodechar{✅}{\checkmark}
\newunicodechar{❌}{\texttimes}
\newunicodechar{🚧}{\textbf{[WIP]}}
\newunicodechar{✗}{\texttimes}
\newunicodechar{≤}{$\leq$}
\newunicodechar{≥}{$\geq$}
\newunicodechar{≈}{$\approx$}
\newunicodechar{↔}{$\leftrightarrow$}
\newunicodechar{→}{$\rightarrow$}
\newunicodechar{←}{$\leftarrow$}
\newunicodechar{—}{---}
\newunicodechar{–}{--}
\newunicodechar{×}{$\times$}
\newunicodechar{├}{|}
\newunicodechar{└}{|}
\newunicodechar{│}{|}
\newunicodechar{─}{-}
\newunicodechar{⚠}{\textbf{!}}
\newunicodechar{🔨}{\textbf{[WIP]}}
\newunicodechar{📋}{\textbf{[PLANNED]}}
\newunicodechar{📅}{\textbf{[FUTURE]}}
\newunicodechar{➖}{\textbf{[-]}}
\DeclareUnicodeCharacter{FE0F}{}
""",
}
