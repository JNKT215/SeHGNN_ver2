#tuning (normal)
IFS_BACKUP=$IFS
IFS=$'\n'

dataset=$1

if [ $dataset = "DBLP" ]; then
    num_hop=2
    num_label_hops=4
    submetapath_hops=2
elif [ $dataset = "IMDB" ]; then
    num_hop=4
    num_label_hops=4
    submetapath_hops=2
    calc_type=linear
    neighbor_aggr_mode=all
    neighbor_encoder=mean
    sampling_limit=10
    submetapath_feature_weight=0.2


elif [ $dataset = "ACM" ]; then
    num_hop=4
    num_label_hops=4
    submetapath_hops=2
fi

echo $num_hop
echo $num_label_hops
echo $submetapath_hops
echo $calc_type
echo $neighbor_aggr_mode
echo $neighbor_encoder
echo $sampling_limit
echo $submetapath_feature_weight



ary=("
     python3 hgb/train.py -m 'key=SeHGNNver2_$dataset' \
     'experiment_name=$dataset tuning' \
     'SeHGNNver2_$dataset.dropout=choice(0, 0.2, 0.4, 0.5, 0.6, 0.8)' \
     'SeHGNNver2_$dataset.num_hop=$num_hop' \
     'SeHGNNver2_$dataset.num_label_hops=$num_label_hops' \
     'SeHGNNver2_$dataset.label_feats=True' \
     'SeHGNNver2_$dataset.residual=True' \
     'SeHGNNver2_$dataset.amp=True' \
     'SeHGNNver2_$dataset.epochs=100' \
     'SeHGNNver2_$dataset.run=5' \
     'SeHGNNver2_$dataset.embed_size=choice(256, 512)' \
     'SeHGNNver2_$dataset.input_drop=choice(0, 0.2, 0.4, 0.6, 0.8)' \
     'SeHGNNver2_$dataset.n_fp_layers=choice(2, 3, 4)' \
     'SeHGNNver2_$dataset.n_task_layers=choice(1, 2, 3, 4)' \
     'SeHGNNver2_$dataset.act=choice(none, relu, leaky_relu, sigmoid)' \
     'SeHGNNver2_$dataset.submetapath_hops=$submetapath_hops' \
     'SeHGNNver2_$dataset.calc_type=$calc_type' \
     'SeHGNNver2_$dataset.neighbor_aggr_mode=$neighbor_aggr_mode'\
     'SeHGNNver2_$dataset.neighbor_encoder=$neighbor_encoder' \
     'SeHGNNver2_$dataset.sampling_limit=choice(10, 15)' \
     'SeHGNNver2_$dataset.submetapath_feature_weight=$submetapath_feature_weight' \
     ")




#  'SeHGNNver2_$dataset.lr=choice(0.01 ,5e-06 ,5e-05 ,0.0005 ,0.005 ,0.05)' \
#  'SeHGNNver2_$dataset.weight_decay=choice(2e-06 ,2e-05 ,2e-04 ,0.002 ,5e-06 ,5e-05 ,5e-04 ,0.005)' \

for STR in ${ary[@]}
do
    eval "${STR}"
done


# source tuning.sh DBLP
# source tuning.sh ACM
# source tuning.sh IMDB
 

 