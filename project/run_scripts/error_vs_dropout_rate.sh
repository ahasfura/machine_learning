#!/bin/bash

me=`basename "$0"`

array=( 0.5 0.6 0.7 0.8 0.9 1.0 )

for i in "${array[@]}"; do
	python ../tensor_flow_nets/katynet.py --dropout_rate $i --run_name $me;
done

