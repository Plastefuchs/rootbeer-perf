#!/usr/bin/env sh

find -maxdepth 1 -name \*.csv -exec ./removeHead.sh {} \;

ls *.csv | sort -R | head -n 1 | xargs head -n 1 > all.csv

cat *_data >> all.csv

rm *.csv_data