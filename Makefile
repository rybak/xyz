TEXC=pdflatex -shell-escape

slides.pdf : slides.tex x.png y.png z.png
	$(TEXC) slides.tex
	$(TEXC) slides.tex

%.png : %.bmp
	convert $^ $@

pictures: rw.py
	python3 rw.py 4 0.99 200
	python3 rw.py 4 0.99 2000
	python3 rw.py 4 0.99 20000

clean :
	rm -f *.aux *.log
