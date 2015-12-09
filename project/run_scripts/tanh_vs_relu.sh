#!/bin/bash
me=`basename "$0"`
python ../tensor_flow_nets/katynet.py --relu False --run_name $me;
python ../tensor_flow_nets/katynet.py --relu True --run_name $me;
