exe (){
     echo python3 $path experiment_name=$experiment_name key=$key $key.epochs=$epochs
          python3 $path experiment_name=$experiment_name key=$key $key.epochs=$epochs
}

add_path=.
# -------SeHGNN run script (exist study)-------
path=${add_path}/hgb/train.py
experiment_name=SeHGNN
epochs=200
#DBLP
key=SeHGNN_DBLP_tuned
exe
echo
#IMDB
key=SeHGNN_IMDB_tuned
exe
echo
#ACM
key=SeHGNN_ACM_tuned
exe
echo

# -------SeHGNNver2 run script (our study)-------
path=${add_path}/hgb/train.py
experiment_name=SeHGNNver2
epochs=50
#DBLP(basic)
key=SeHGNNver2_DBLP_tuned_basic_alpha_manual
exe
echo

key=SeHGNNver2_DBLP_tuned_basic_alpha_auto
exe
echo

key=SeHGNNver2_DBLP_tuned_basic_transformer
exe
echo

#IMDB(basic)
key=SeHGNNver2_IMDB_tuned_basic_alpha_manual
exe
echo

key=SeHGNNver2_IMDB_tuned_basic_alpha_auto
exe
echo

key=SeHGNNver2_IMDB_tuned_basic_transformer
exe
echo

#ACM(basic)
key=SeHGNNver2_ACM_tuned_basic_alpha_manual
exe
echo

key=SeHGNNver2_ACM_tuned_basic_alpha_auto
exe
echo

key=SeHGNNver2_ACM_tuned_basic_transformer
exe
echo

#concat
#DBLP(concat)
key=SeHGNNver2_DBLP_tuned_concat_alpha_manual
exe
echo

key=SeHGNNver2_DBLP_tuned_concat_alpha_auto
exe
echo

key=SeHGNNver2_DBLP_tuned_concat_transformer
exe
echo

#IMDB(concat)
key=SeHGNNver2_IMDB_tuned_concat_alpha_manual
exe
echo

key=SeHGNNver2_IMDB_tuned_concat_alpha_auto
exe
echo

key=SeHGNNver2_IMDB_tuned_concat_transformer
exe
echo

#ACM(concat)
key=SeHGNNver2_ACM_tuned_concat_alpha_manual
exe
echo

key=SeHGNNver2_ACM_tuned_concat_alpha_auto
exe
echo

key=SeHGNNver2_ACM_tuned_concat_transformer
exe
echo

