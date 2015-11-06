#!/usr/bin/env sh

pi_jar="java -jar pi/dist/pi-gpu.jar "
sort_jar="java -jar sort/dist/sorter-gpu.jar "

# pi sampling

pi_1="512 14 512 10 1000"
pi_2="512 14 512 10 10000"
pi_3="512 14 512 10 100000"
# pi_4="512 14 512 10 1000000"
# pi_5="512 14 512 10 10000000"

eval "$pi_jar $pi_1"
eval "$pi_jar $pi_2"
eval "$pi_jar $pi_3"
eval "$pi_jar $pi_4"
eval "$pi_jar $pi_5"

pi_1="1024 14 512 10 1000"
pi_2="1024 14 512 10 10000"
pi_3="1024 14 512 10 100000"
# pi_4="1024 14 512 10 1000000"
# pi_5="1024 14 512 10 10000000"

eval "$pi_jar $pi_1"
eval "$pi_jar $pi_2"
eval "$pi_jar $pi_3"
eval "$pi_jar $pi_4"
eval "$pi_jar $pi_5"

./csv_helper.sh
mv all.csv outcome/all_pi.csv
mv *.csv outcome/pi/

# sorting

sort_1="512 1 512 10"
sort_2="512 2 512 10"
sort_3="512 4 512 10"
sort_4="512 8 512 10"
sort_5="512 14 512 10"

eval "$sort_jar $sort_1"
eval "$sort_jar $sort_2"
# eval "$sort_jar $sort_3"
# eval "$sort_jar $sort_4"
# eval "$sort_jar $sort_5"

sort_1="512 14 32 10"
sort_2="512 14 64 10"
sort_3="512 14 128 10"
sort_4="512 14 256 10"
sort_5="512 14 512 10"

eval "$sort_jar $sort_1"
eval "$sort_jar $sort_2"
# eval "$sort_jar $sort_3"
# eval "$sort_jar $sort_4"
# eval "$sort_jar $sort_5"


sort_1="128 14 512 10"
sort_2="256 14 512 10"
sort_3="512 14 512 10"
sort_4="1024 14 512 10"
sort_5="2048 14 512 10"

eval "$sort_jar $sort_1"
eval "$sort_jar $sort_2"
# eval "$sort_jar $sort_3"
# eval "$sort_jar $sort_4"
# eval "$sort_jar $sort_5"


./csv_helper.sh
mv all.csv  outcome/all_sort.csv
mv *.csv outcome/sort/