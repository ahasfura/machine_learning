#!/bin/bash

me=`basename "$0"`

array=( 2 4 6 8 )

for i in "${array[@]}"; do
	python ../tensor_flow_nets/katynet.py --pool_dim $i --run_name $me;
done

