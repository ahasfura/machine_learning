run_breaker="%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
for i in {1..20}; do
  python ../tensor_flow_nets/lenet1.py >> ../run_outputs/run_all_nets.txt;
  echo $run_breaker >> ../run_outputs/run_all_nets.txt;
  python ../tensor_flow_nets/lenet4.py >> ../run_outputs/run_all_nets.txt;
  echo $run_breaker >> ../run_outputs/run_all_nets.txt;
  python ../tensor_flow_nets/lenet5.py >> ../run_outputs/run_all_nets.txt;
  echo $run_breaker >> ../run_outputs/run_all_nets.txt;
  python ../tensor_flow_nets/three_layer_fewer_units.py >> ../run_outputs/run_all_nets.txt;
  echo $run_breaker >> ../run_outputs/run_all_nets.txt;
  python ../tensor_flow_nets/three_layer_more_units.py >> ../run_outputs/run_all_nets.txt;
  echo $run_breaker >> ../run_outputs/run_all_nets.txt;
  python ../tensor_flow_nets/two_layer_1000_units.py >> ../run_outputs/run_all_nets.txt;
  echo $run_breaker >> ../run_outputs/run_all_nets.txt;
  python ../tensor_flow_nets/two_layer_300_units.py >> ../run_outputs/run_all_nets.txt;
  echo $run_breaker >> ../run_outputs/run_all_nets.txt;
done
