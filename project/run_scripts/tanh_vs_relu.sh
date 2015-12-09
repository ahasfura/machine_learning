#!/bin/bash
me=`basename "$0"`
python ../tensor_flow_nets/katynet.py --relu --run_name $me --training_iters 100000;
python ../tensor_flow_nets/katynet.py --tanh --run_name $me --training_iters 100000;
