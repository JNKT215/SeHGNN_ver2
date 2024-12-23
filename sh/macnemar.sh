macnemar_exe(){
    for seed in 1 2 3 4 5
    do
        SeHGNN_dir=SeHGNN_${dataset}
        SeHGNNver2_dir=SeHGNNver2_${dataset}
        SeHGNN_file_path=${add_path}/${SeHGNN_dir}/${dataset}_${seed}_macnemar_${dataset}_checkpoint.txt
        SeHGNNver2_file_path=${add_path}/${SeHGNNver2_dir}/${dataset}_${seed}_macnemar_${dataset}_checkpoint.txt

        echo python3 $path --dataset $dataset --seed $seed --SeHGNN $SeHGNN_file_path --SeHGNN_ver2 $SeHGNNver2_file_path
             python3 $path --dataset $dataset --seed $seed --SeHGNN $SeHGNN_file_path --SeHGNN_ver2 $SeHGNNver2_file_path
    done
}
#input_data_path
add_path=.
#py_file_path
path=debug/macnemar_test.py

#DBLP
dataset=DBLP
macnemar_exe
echo

#IMDB
dataset=IMDB
macnemar_exe
echo

#ACM
dataset=ACM
macnemar_exe
echo