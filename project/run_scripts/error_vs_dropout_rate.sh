#!/bin/bash

me=`basename "$0"`

array=( 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 )

for i in "${array[@]}"; do
	python ../tensor_flow_nets/katynet.py --dropout_rate $i --run_name $me;
done

