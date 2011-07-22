# Simple makefile

PYTHON ?= python
PYLINT_WRAPPER ?= ./py_lintw.sh
NOSETESTS ?= nosetests

all: clean build test

build:
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -rf build/
	rm -rf src/build/
	find . -name "*.pyc" | xargs rm -f
	find . -name "*.c" | xargs rm -f
	find . -name "*.so" | xargs rm -f

test: 
	$(NOSETESTS)

lint: 
	$(PYLINT_WRAPPER) tagassess

detlint: 
	$(PYLINT_WRAPPER) -r y tagassess
	
trailing-spaces: 
	find -name "*.py" | xargs sed 's/^M$$//'
