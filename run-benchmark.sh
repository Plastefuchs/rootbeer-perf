#!/usr/bin/env bash

pi_jar="java -jar pi/dist/pi-gpu.jar "
sort_jar="java -jar sort/dist/sorter-gpu.jar "

# pi sampling

iterations=(1000 10000 100000 1000000 10000000)
threads=(512 1024)
mp_number=(1 2 4 8 14)

for thread in ${threads[@]}
do
	for iter in ${iterations[@]}
	do
		for mp in ${mp_number[@]}
		do
			eval "$pi_jar $thread $mp 512 10 $iter"
		done
	done
done

./csv_helper.sh
timestamp=`date +%F_%H%M%S`
outcome_pi="outcome/pi_$timestamp"
mkdir $outcome_pi
mv all.csv outcome/${timestamp}_all_pi.csv
mv *.csv $outcome_pi

# sorting

mp_number=(1 2 4 8 14)
threads=(128 256 512 1024 2048)
blocks=(32 64 128 256 512)

for mp in ${mp_number[@]}
do
	for thread in ${threads[@]}
	do
		for block in ${blocks[@]}
		do
			eval "$sort_jar $thread $mp $block 10"
		done
	done
done

./csv_helper.sh
timestamp=`date +%F_%H%M%S`
outcome_sort="outcome/sort_$timestamp"
mkdir $outcome_sort
mv all.csv outcome/${timestamp}_all_sort.csv
mv *.csv $outcome_sort