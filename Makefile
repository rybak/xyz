TEXC=pdflatex -shell-escape

slides.pdf : slides.tex x.png y.png z.png
	$(TEXC) slides.tex
	$(TEXC) slides.tex

%.png : %.bmp
	convert $^ $@

clean :
	rm -f *.aux *.log
