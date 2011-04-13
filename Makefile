# Simple makefile

PYTHON ?= python
PYLINT_WRAPPER ?= ./py_lintw.sh
NOSETESTS ?= nosetests

all: clean test

clean:
	find . -name "*.pyc" | xargs rm -f

test:
	$(NOSETESTS)

lint:
	$(PYLINT_WRAPPER) tagassess

detlint:
	$(PYLINT_WRAPPER) -r y tagassess
	
trailing-spaces:
	find -name "*.py" | xargs sed 's/^M$$//'
	find -name "*.py" | xargs sed -i 's/[ \t]*$$//'