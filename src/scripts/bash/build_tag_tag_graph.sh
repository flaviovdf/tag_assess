#!/bin/bash
# 
# Computes a tag-tag graph based on tag co-occurrence on items. The script 
# receives as input the sql script that finds the edges, the database and the
# table names. If the user wants to use a different db server, she needs to
# change the MYSQL_CMD variable below. 
#
# $1 -- the sql script
# $2 -- the database name
# $2 -- the table name
#
if [ $# -ne 3 ]; then
    echo "Usage: " $0 "<sql script> <db> <table>"
    exit 1
fi

MYSQL_CMD="$HOME/software/mysql/bin/mysql -N --host=10.0.0.230 --port=33600 --database=$2 --user=root"

for V in `echo "SELECT DISTINCT(taginfo_id) as tag FROM tas" | $MYSQL_CMD`; do
  sed 's/\#1/'$3'/' $1 | sed 's/\#2/'$V'/' | $MYSQL_CMD | awk '{print v, $2, $1}' v=$V
done
