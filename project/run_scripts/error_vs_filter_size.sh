#!/bin/bash
me=`basename "$0"`
for i in {2..15}; do
	python ../tensor_flow_nets/katynet.py --filter_size $i --run_name $me;
done

