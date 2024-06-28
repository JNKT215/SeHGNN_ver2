import torch
import torch.nn.functional as F
import numpy as np
import numpy.linalg as LA
import os
import mlflow
import random
import uuid
import sys
import pickle,joblib,csv

def to_homogeneous_edge_index(data):
    # Record slice information per node type:
    cumsum = 0
    node_slices = {}
    for node_type, store in data._node_store_dict.items():
        num_nodes = store.num_nodes
        node_slices[node_type] = (cumsum, cumsum + num_nodes)
        cumsum += num_nodes

    # Record edge indices and slice information per edge type:
    cumsum = 0
    edge_indices = []
    edge_slices = {}
    for edge_type, store in data._edge_store_dict.items():
        src, _, dst = edge_type
        offset = [[node_slices[src][0]], [node_slices[dst][0]]]
        offset = torch.tensor(offset, device=store.edge_index.device)
        edge_indices.append(store.edge_index + offset)

        num_edges = store.num_edges
        edge_slices[edge_type] = (cumsum, cumsum + num_edges)
        cumsum += num_edges

    edge_index = None
    if len(edge_indices) == 1:  # Memory-efficient `torch.cat`:
        edge_index = edge_indices[0]
    elif len(edge_indices) > 0:
        edge_index = torch.cat(edge_indices, dim=-1)

    return edge_index, node_slices, edge_slices

class EarlyStopping():
    def __init__(self,dataset,patience,path="checkpoint.pt"):
        self.dataset = dataset
        self.best_val_loss = None
        self.loss_counter =0
        self.patience = patience
        self.path = path
        self.val_loss_min =None
        self.best_epoch = 0
        
        # only need My-theis
        self.best_test_loss = None
        self.best_val = None
        self.best_test =None
        self.best_raw_preds = None
        
    def __call__(self,model,epoch,log,val_loss,test_loss,val_acc,test_acc,raw_preds):
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_best_model(model,val_loss)
            self.best_epoch = epoch
            
            #only need my-theis
            self.best_test_loss = test_loss
            self.best_val = val_acc        
            self.best_test =test_acc
            self.best_raw_preds = raw_preds
            
        elif (self.dataset!= 'Freebase' and self.best_val_loss > val_loss) or (self.dataset == "Freebase" and sum(val_acc) > sum(self.best_val)):
            self.best_val_loss = val_loss
            self.loss_counter = 0
            self.save_best_model(model,val_loss)
            self.best_epoch = epoch
            
            #only need my-theis
            self.best_test_loss = test_loss
            self.best_val = val_acc        
            self.best_test =test_acc
            self.best_raw_preds = raw_preds
        else:
            self.loss_counter+=1
        
        if epoch > 0 and epoch % 10 == 0: 
            log = log + f'\tCurrent best at epoch {self.best_epoch} with Val loss {self.best_val_loss:.4f} ({self.best_val[0]*100:.4f}, {self.best_val[1]*100:.4f})' \
                + f', Test loss {self.best_test_loss:.4f} ({self.best_test[0]*100:.4f}, {self.best_test[1]*100:.4f})'
        
        print(log)
            
        if self.loss_counter == self.patience:
            return True
        
        return False
    def save_best_model(self,model,val_loss):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
    
    def print_best_epoch_result(self):
        print(f'Best Epoch {self.best_epoch} at {self.path[:-3] }\n\tFinal Val loss {self.best_val_loss:.4f} ({self.best_val[0]*100:.4f}, {self.best_val[1]*100:.4f})'
                    + f', Test loss {self.best_test_loss:.4f} ({self.best_test[0]*100:.4f}, {self.best_test[1]*100:.4f})')
        

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def log_artifacts(artifacts,output_path=None):
    if artifacts is not None:
        for artifact_name, artifact in artifacts.items():
            if isinstance(artifact, list):
                if output_path is not None:
                    artifact_name = f"{output_path}/{artifact_name}"
                    os.makedirs(output_path, exist_ok=True)
                np.save(artifact_name, artifact)
                mlflow.log_artifact(artifact_name)
            elif artifact is not None and artifact !=[]:
                if output_path is not None:
                    artifact_name = f"{output_path}/{artifact_name}"
                    os.makedirs(output_path, exist_ok=True)
                np.save(artifact_name, artifact.to('cpu').detach().numpy().copy())
                mlflow.log_artifact(artifact_name)
                

def set_checkpt_folder(cfg):
    checkpt_folder = f'./output/{cfg.dataset}/'
    if not os.path.exists(checkpt_folder):
        os.makedirs(checkpt_folder)
    checkpt_file = checkpt_folder + uuid.uuid4().hex
    print('checkpt_file', checkpt_file)

def save_gs_features(gs_pt_folder_path,gs_file_name,gs_feature,exit_py=False):
    if not os.path.exists(gs_pt_folder_path):
        os.makedirs(gs_pt_folder_path)
    torch.save(gs_feature,f'{gs_pt_folder_path}/{gs_file_name}')
    if exit_py:
        sys.exit()
                

#リスト型からtensor型に変換するときの高速化
def convert_list_to_tensor(x,device,dtype): 
    return torch.tensor(np.array(x, dtype=dtype), device=device)

def check_acc(preds_dict, condition, init_labels, train_nid, val_nid, test_nid, show_test=True, loss_type='ce'):
    mask_train, mask_val, mask_test = [], [], []
    remove_label_keys = []
    k = list(preds_dict.keys())[0]
    v = preds_dict[k]
    if loss_type == 'ce':
        na, nb, nc = len(train_nid), len(val_nid), len(test_nid)
    elif loss_type == 'bce':
        na, nb, nc = len(train_nid) * v.size(1), len(val_nid) * v.size(1), len(test_nid) * v.size(1)

    for k, v in preds_dict.items():
        if loss_type == 'ce':
            pred = v.argmax(1)
        elif loss_type == 'bce':
            pred = (v > 0).int()

        a, b, c = pred[train_nid] == init_labels[train_nid], \
                  pred[val_nid] == init_labels[val_nid], \
                  pred[test_nid] == init_labels[test_nid]
        ra, rb, rc = a.sum() / na, b.sum() / nb, c.sum() / nc

        if loss_type == 'ce':
            vv = torch.log(v / (v.sum(1, keepdim=True) + 1e-6) + 1e-6)
            la, lb, lc = F.nll_loss(vv[train_nid], init_labels[train_nid]), \
                         F.nll_loss(vv[val_nid], init_labels[val_nid]), \
                         F.nll_loss(vv[test_nid], init_labels[test_nid])
        else:
            vv = (v / 2. + 0.5).clamp(1e-6, 1-1e-6)
            la, lb, lc = F.binary_cross_entropy(vv[train_nid], init_labels[train_nid].float()), \
                         F.binary_cross_entropy(vv[val_nid], init_labels[val_nid].float()), \
                         F.binary_cross_entropy(vv[test_nid], init_labels[test_nid].float())
        if condition(ra, rb, rc, k):
            mask_train.append(a)
            mask_val.append(b)
            mask_test.append(c)
        else:
            remove_label_keys.append(k)
        if show_test:
            print(k, ra, rb, rc, la, lb, lc, (ra/rb-1)*100, (ra/rc-1)*100, (1-la/lb)*100, (1-la/lc)*100)
        else:
            print(k, ra, rb, la, lb, (ra/rb-1)*100, (1-la/lb)*100)
    print(set(list(preds_dict.keys())) - set(remove_label_keys))

    print((torch.stack(mask_train, dim=0).sum(0) > 0).sum() / na)
    print((torch.stack(mask_val, dim=0).sum(0) > 0).sum() / nb)
    if show_test:
        print((torch.stack(mask_test, dim=0).sum(0) > 0).sum() / nc)
        
        
        
def enum_metapath_name(cfg,name_dict, type_dict, length):
    hop = []
    path_list = []
    result_dict = {}
    for type in type_dict.keys():
        hop.append([type])
        result_dict[name_dict[type][0]] = []
    path_list.extend(hop)
    for i in range(length - 2):
        new_hop = []
        for path in hop:
            for next_type in type_dict[path[-1]]:
                new_hop.append(path + [next_type])
        hop = new_hop
        path_list.extend(hop)
    for path in path_list:
        name = name_dict[path[0]][0]
        for index in path:
            name += name_dict[index][1]
        if len(name) > 1:
            result_dict[name[0]].append(name)
    
    # if cfg['only_tgt_type']:
    #     result_list = [metapath  for metapath in result_dict[cfg['tgt_type']]]
    #     return result_list
    
    return result_dict


def enum_all_metapath(cfg,name_dict, type_dict, length,metapath_names):
    hop = []
    path_list = []
    for type in type_dict.keys():
        hop.append([type])
    path_list.extend(hop)
    for i in range(length - 2):
        new_hop = []
        for path in hop:
            for next_type in type_dict[path[-1]]:
                new_hop.append(path + [next_type])
        hop = new_hop
        path_list.extend(hop)
    path_dict = {}
    for path in path_list:
        name = name_dict[path[0]][0]
        for index in path:
            name += name_dict[index][1]
        path_dict[name] = path
    
    # if cfg['only_tgt_type']:
    #     path_dict = {metapath:path_dict[metapath] for metapath in metapath_names}
    #     return path_dict    
    
    return path_dict


def save_and_load_dict_pkl(file_path,dict_path,save_dict_variable=None):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if os.path.isfile(f"{file_path}/{dict_path}"):
        with open(f"{file_path}/{dict_path}", 'rb') as f:
            save_dict_variable = pickle.load(f)
    else:
        if save_dict_variable is not None:
            with open(f"{file_path}/{dict_path}","wb") as f:
                # pickle.dump(save_dict_variable, f)
                pickle.dump(obj=save_dict_variable,file=f,protocol=4)
    return save_dict_variable





def save_and_load_tensor_pt(file_path,dict_path,save_dict_variable=None):
    pass


def write_SeHGNN_epoch_to_loss_to_time(dir_path,output_file_name,model_name,repletion,epochs,times,losses):
    os.makedirs(dir_path, exist_ok=True)
    # ファイルを書き込みモードで開く
    with open(f"{dir_path}/{output_file_name}", mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # 最初の行にモデル名を追加
        writer.writerow([model_name])
        
        # エポックの見出し行を追加
        epoch_header = [""] + [f"{epoch}epoch" for epoch in epochs]
        writer.writerow(epoch_header)
        
        # Loss の行を追加
        loss_row = [f"Loss_{repletion}"] + losses
        writer.writerow(loss_row)
        
        # Time の行を追加
        time_row = [f"Time_{repletion}"] + times
        writer.writerow(time_row)

    print(f"{output_file_name} に書き込みが完了しました。")


def concat_output_csv_file_for_write_epoch_to_loss_to_time_format(dir_path,output_file_name,input_files):
# 出力ファイル名
    

    # 初期化
    model_name = ""
    epoch_header = []
    loss_rows = []
    time_rows = []

    # 各ファイルを読み込んでデータを集める
    for i, file_name in enumerate(input_files):
        with open(f"{dir_path}/{file_name}", mode='r') as file:
            reader = csv.reader(file)
            lines = list(reader)
            
            if i == 0:
                # モデル名とエポックのヘッダーを取得
                model_name = lines[0][0]
                epoch_header = lines[1]
            
            # Loss 行を取得して連結
            loss_row = lines[2]
            # loss_row[0] = f"Loss（ファイル{i+1}）"
            loss_rows.append(loss_row)
            
            # Time 行を取得して連結
            time_row = lines[3]
            # time_row[0] = f"Time（ファイル{i+1}）"
            time_rows.append(time_row)

    # 出力ファイルに書き込む
    with open(f"{dir_path}/{output_file_name}", mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # モデル名を書き込む
        writer.writerow([model_name])
        
        # エポックのヘッダーを書き込む
        writer.writerow(epoch_header)
        
        # Loss の行を書き込む
        for row in loss_rows:
            writer.writerow(row)
        
        # モデル名とエポックのヘッダーを書き込む
        writer.writerow([])
        writer.writerow([model_name])
        writer.writerow(epoch_header)
        
        # Time の行を書き込む
        for row in time_rows:
            writer.writerow(row)

    print(f"{output_file_name} に連結が完了しました。")