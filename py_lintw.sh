#!/bin/bash
#Script that configures the PYTHONPATH variable so that we can run pylint.

findself() {
    SELF=`dirname $0`
}

findself

PYTHONPATH=$PYTHONPATH:$SELF/src pylint --rcfile=$SELF/pylint.rc $*
