# Simple makefile

SOURCEDIR = src/
PYTHON ?= python
NOSETESTS ?= nosetests --exe
CTAGS ?= ctags

all: clean test

clean:
	find . -name "*.pyc" | xargs rm -f

test:
	$(NOSETESTS)
	
trailing-spaces:
	find -name "*.py" |xargs sed -i 's/[ \t]*$$//'
