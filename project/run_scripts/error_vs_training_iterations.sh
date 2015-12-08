#!/bin/bash
me=`basename "$0"`
for i in {1..4}; do
	python ../tensor_flow_nets/katynet.py --run_name $me --dont_save_run > ../run_outputs/$me.txt --training_iters 150000;
done

