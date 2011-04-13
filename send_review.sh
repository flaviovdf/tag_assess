#!/bin/bash 
# Simple script to submit Rietveld code reviews.

findself() {
    SELF=`dirname $0`
}

findself

cd $SELF
make clean
cd -

python $SELF/upload.py --reviewers=elsantosneto@gmail.com --do --send_mail --private --rev=HEAD -s http://codereview-netsyslab.appspot.com $*
