#tuning (normal)
IFS_BACKUP=$IFS
IFS=$'\n'
dataset=$1
gpu_id=$2
if [ $dataset = "DBLP" ]; then
    num_hop=2
    num_label_hops=4
    submetapath_hops=2
    neighbor_aggr_calc=basic
    neighbor_aggr_mode=transformer
elif [ $dataset = "IMDB" ]; then
    num_hop=4
    num_label_hops=4
    submetapath_hops=2
    neighbor_aggr_calc=basic
    neighbor_aggr_mode=transformer
    elif [ $dataset = "ACM" ]; then
    num_hop=4
    num_label_hops=4
    submetapath_hops=2
    neighbor_aggr_calc=basic
    neighbor_aggr_mode=transformer
fi
echo gpu_id=$gpu_id
echo num_hop=$num_hop
echo num_label_hops=$num_label_hops
echo submetapath_hops=$submetapath_hops
echo neighbor_aggr_calc=$neighbor_aggr_calc
echo neighbor_aggr_mode=$neighbor_aggr_mode
ary=("
     python3 hgb/train.py -m 'key=SeHGNNver2_$dataset' \
     'experiment_name=tuning' \
     'SeHGNNver2_$dataset.label_feats=True' \
     'SeHGNNver2_$dataset.amp=True' \
     'SeHGNNver2_$dataset.epochs=50' \
     'SeHGNNver2_$dataset.run=5' \
     'SeHGNNver2_$dataset.embed_size=choice(256, 512)' \
     'SeHGNNver2_$dataset.lr=choice(1e-3, 1e-2, 5e-3, 5e-2)' \
     'SeHGNNver2_$dataset.weight_decay=choice(0, 1e-4,  5e-4, 1e-3)' \
     'SeHGNNver2_$dataset.dropout=choice(0, 0.2, 0.4, 0.5)' \
     'SeHGNNver2_$dataset.input_drop=choice(0, 0.2, 0.4, 0.5)' \
     'SeHGNNver2_$dataset.n_task_layers=choice(1, 2, 3, 4)' \
     'SeHGNNver2_$dataset.gpu_id=$gpu_id' \
     'SeHGNNver2_$dataset.num_hop=$num_hop' \
     'SeHGNNver2_$dataset.num_label_hops=$num_label_hops' \
     'SeHGNNver2_$dataset.submetapath_hops=$submetapath_hops' \
     'SeHGNNver2_$dataset.neighbor_aggr_calc=$neighbor_aggr_calc' \
     'SeHGNNver2_$dataset.neighbor_aggr_mode=$neighbor_aggr_mode'\
     ")
for STR in ${ary[@]}
do
    eval "${STR}"
done