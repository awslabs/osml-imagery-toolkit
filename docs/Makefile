# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Convenience target: build PDF via LaTeX
# Requires: pdflatex and latexmk (install via `brew install --cask mactex-no-gui`)
pdf:
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "PDF available at $(BUILDDIR)/latex/osml-imagery-toolkit.pdf"

# Convenience target: build only the User Guide PDF
pdf-user-guide:
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "User Guide PDF available at $(BUILDDIR)/latex/osml-imagery-toolkit-user-guide.pdf"

.PHONY: help pdf pdf-user-guide Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
