edm:
	pandoc -s --bibliography=biblio.bib --biblatex presentation.md -t beamer -o presentation.tex
	pdflatex presentation
	biber presentation
	pdflatex presentation
	open presentation.pdf

edm_simple:
	pandoc presentation.md -t beamer -o presentation.pdf
	open presentation.pdf

jdse:
	pandoc -s --bibliography=biblio.bib --biblatex jdse.md -t beamer -o jdse.tex
	pdflatex jdse
	biber jdse
	pdflatex jdse
	open jdse.pdf

jdse_simple:
	pandoc jdse.md -t beamer -o jdse.pdf
	open jdse.pdf

limsi:
	pandoc -s --bibliography=biblio.bib --biblatex limsi.md -t beamer -o limsi.tex
	pdflatex limsi
	biber limsi
	pdflatex limsi
	open limsi.pdf

limsi_simple:
	pandoc limsi.md -t beamer -o limsi.pdf
	open limsi.pdf

lak:
	pandoc -s --bibliography=biblio.bib --biblatex lak2020.md -t beamer -o lak2020.tex
	pdflatex lak2020
	biber lak2020
	pdflatex lak2020
	open lak2020.pdf

lak_simple:
	pandoc lak2020.md -t beamer -o lak2020.pdf
	open lak2020.pdf

lip6:
	pandoc -s --bibliography=biblio.bib --biblatex lip6.md -t beamer -o lip6.tex
	pdflatex lip6
	biber lip6
	pdflatex lip6
	open lip6.pdf

lip6_simple:
	pandoc lip6.md -t beamer -o lip6.pdf
	open lip6.pdf

clean:
	rm -f *.aux *.bbl *.bcf *.blg *.log *.nav *.out *.run.xml *.snm *.synctex \
		*.synctex.gz *.toc *.vrb
	rm -f */*.log */*.aux