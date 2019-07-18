edm:
	pandoc -s --bibliography=biblio.bib --biblatex presentation.md -t beamer -o presentation.tex
	pdflatex presentation
	biber presentation
	pdflatex presentation
	open presentation.pdf

edm_simple:
	pandoc presentation.md -t beamer -o presentation.pdf
	open presentation.pdf