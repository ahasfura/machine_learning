#!/bin/bash

#array=( 6875 13750 20625 27500 34375 41250 48125 55000 )
<<<<<<< Updated upstream
array=( 256 500 1000 2000 4000 5000 10000 20000 30000 40000 55000 )
=======

array=( 55000 )
>>>>>>> Stashed changes

for i in "${array[@]}"; do
	python ../tensor_flow_nets/lenet5.py $i;
done

