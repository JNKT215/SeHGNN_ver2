# -------SeHGNN run script (exist study)-------
#DBLP
path=hgb/train.py
experiment_name=SeHGNN
key=SeHGNN_DBLP_tuned
echo python3 $path experiment_name=$experiment_name key=$key 
     python3 $path experiment_name=$experiment_name key=$key
#IMDB
path=hgb/train.py
experiment_name=SeHGNN
key=SeHGNN_IMDB_tuned
echo python3 $path experiment_name=$experiment_name key=$key 
     python3 $path experiment_name=$experiment_name key=$key
#ACM
path=hgb/train.py
experiment_name=SeHGNN
key=SeHGNN_ACM_tuned
echo python3 $path experiment_name=$experiment_name key=$key 
     python3 $path experiment_name=$experiment_name key=$key

# -------SeHGNNver2 run script (our study)-------
#DBLP
path=hgb/train.py
experiment_name=SeHGNN
key=SeHGNNver2_DBLP_tuned
echo python3 $path experiment_name=$experiment_name key=$key 
     python3 $path experiment_name=$experiment_name key=$key
#IMDB
path=hgb/train.py
experiment_name=SeHGNN
key=SeHGNNver2_IMDB_tuned
echo python3 $path experiment_name=$experiment_name key=$key 
     python3 $path experiment_name=$experiment_name key=$key
#ACM
path=hgb/train.py
experiment_name=SeHGNN
key=SeHGNNver2_ACM_tuned
echo python3 $path experiment_name=$experiment_name key=$key 
     python3 $path experiment_name=$experiment_name key=$key

