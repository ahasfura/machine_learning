#!/bin/bash

array=( 6875 13750 20625 27500 34375 41250 48125 55000 )

for i in "${array[@]}"; do
	python ../tensor_flow_nets/lenet1.py $i;
done

