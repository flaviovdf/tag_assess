#!/bin/bash 
#Script that configures the PYTHONPATH variable so that the interpreter can find our code.

findself() {
    SELF=`dirname $0`
}

findself

PYTHONPATH=$PYTHONPATH:$SELF/src python $*
