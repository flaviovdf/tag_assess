#!/bin/bash
# This script sorts a totem file that does not have edges presented
# in sorted order.
#
# Author - Flavio Figueiredo
# Date - 24/05/2011

if [ $# -lt 2 ]; then
    echo 'Usage $0 <in_file> <out_file>'
    exit 1
fi

IN=$1
OUT=$2

grep '#' $IN > $OUT
grep -v '#' $IN | sort --key=1,1n --key=2n $tmp >> $OUT