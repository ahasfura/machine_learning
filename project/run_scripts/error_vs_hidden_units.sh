array=( 10 100 1000 10000 100000 1000000 )

for i in "${array[@]}"; do
  echo '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%' >> ../run_outputs/error_vs_num_hidden_nodes.txt;
  echo $i >> ../run_outputs/error_vs_num_hidden_nodes.txt;
  python ../tensor_flow_nets/two_layer_units.py $i >> ../run_outputs/error_vs_num_hidden_nodes.txt;
done
