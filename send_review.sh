#!/bin/bash
# Simple script to submit Rietveld code reviews.

findself() {
    SELF=`dirname $0`
}

findself

cd $SELF
make clean
cd -

python $SELF/upload.py --reviewers=flaviovdf@gmail.com,elsantosneto@gmail.com --do --send_mail --private --download_base --rev=HEAD --base_url=https://github.com/flaviovdf/tag_assess.git -s http://codereview-netsyslab.appspot.com $*
