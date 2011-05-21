#!/bin/bash
# 
# Computes the shortest paths from a given source node to all other nodes in the
# graph given as input to the script. The script assumes the file is a .totem
# format file with directed edges list. It also assumes the the totem command
# is located at the current directory.
#
# Author: Elizeu Santos-Neto
# Date: 2011-05-21
#

if [ $# -ne 1 ]; then
    echo "Usage: " $0 "<.totem file>"
    exit 1
fi

TOTEM=~/Dropbox/totem-graph/trunk/src/totem/graph
GRAPH=$1

for S in `grep -v \# $GRAPH | awk '{ print $1}' | uniq`; do
    echo Running source $S ...
    $TOTEM -s $S $GRAPH > sp_${S}.dat
done

echo Done.