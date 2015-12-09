#!/bin/bash

me=`basename "$0"`

array=( 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 )

for i in "${array[@]}"; do
	python ../tensor_flow_nets/katynet.py --training_iters $i --run_name $me;
done

