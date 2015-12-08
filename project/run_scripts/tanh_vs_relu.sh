#!/bin/bash
me=`basename "$0"`
python ../tensor_flow_nets/katynet.py --relu False --run_name $me --training_iters 10000;
python ../tensor_flow_nets/katynet.py --relu True --run_name $me --training_iters 10000;
