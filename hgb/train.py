import torch
import hydra,gc
from hydra import utils
from tqdm import tqdm
import mlflow
from utils import EarlyStopping,set_random_seed,set_checkpt_folder
# from data import HeteroDataLoader
import torch 
import time
from model import PreProcessing,return_model
from sklearn.metrics import f1_score
import torch.nn as nn
import os,sys
from datetime import datetime
sys.path.append('./data')
from data_loader import HeteroDataSet

def evaluator(gt, pred):
    gt = gt.cpu().squeeze()
    pred = pred.cpu().squeeze()
    return f1_score(gt, pred, average='micro'), f1_score(gt, pred, average='macro')


def get_eval_and_full_loader(cfg,data):
    # Freebase train/val/test/full_nodes: 1909/477/5568/40402
    # IMDB     train/val/test/full_nodes: 1097/274/3202/359
    eval_loader, full_loader = [], []
    eval_batch_size = 2 * cfg['batch_size']

    for batch_idx in range((data.labeled_num_nodes-1) // eval_batch_size + 1):
        batch_start = batch_idx * eval_batch_size
        batch_end = min(data.labeled_num_nodes, (batch_idx+1) * eval_batch_size)
        batch = torch.LongTensor(data.labeled_nid[batch_start:batch_end])

        batch_feats = {k: x[batch] for k, x in data.feats.items()}
        batch_labels_feats = {k: x[batch] for k, x in data.label_feats.items()}
        batch_mask = None
        eval_loader.append((batch, batch_feats, batch_labels_feats, batch_mask))

    for batch_idx in range((len(data.extra_nid)-1) // eval_batch_size + 1):
        batch_start = batch_idx * eval_batch_size
        batch_end = min(len(data.extra_nid), (batch_idx+1) * eval_batch_size)
        batch = torch.LongTensor(data.extra_nid[batch_start:batch_end])

        batch_feats = {k: x[batch] for k, x in data.feats.items()}
        batch_labels_feats = {k: x[batch] for k, x in data.label_feats.items()}
        batch_mask = None
        full_loader.append((batch, batch_feats, batch_labels_feats, batch_mask))
    
    return eval_loader, full_loader


def train(model, feats, label_feats, neighbor_aggr_feature_per_metapath, labels_cuda, loss_fcn, optimizer, train_loader, evaluator,mask=None, scalar=None):
    model.train()
    device = labels_cuda.device #本来の奴
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []    
    submetapath_feats = model.submetapath_aggr(neighbor_aggr_feature_per_metapath) if model.cfg["model"] == "SeHGNNver2" else {}
        
    for batch in train_loader:
        # batch = batch.to(device)
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
            batch_submetapath_feats = [x[batch].to(device) for x in submetapath_feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            batch_submetapath_feats = {k: x[batch].to(device) for k, x in submetapath_feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
        batch_y = labels_cuda[batch]

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                if model.cfg["model"] == "SeHGNNver2": 
                    output_att = model(batch, batch_feats, batch_submetapath_feats,batch_labels_feats, batch_mask)
                else:
                    output_att = model(batch, batch_feats, batch_labels_feats, batch_mask)
                    
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            if model.cfg["model"] == "SeHGNNver2": 
                output_att = model(batch, batch_feats, batch_submetapath_feats,batch_labels_feats, batch_mask)
            else:
                output_att = model(batch, batch_feats, batch_labels_feats, batch_mask)
            loss_train = loss_fcn(output_att, batch_y)
            loss_train.backward()
            optimizer.step()

        y_true.append(batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, acc, submetapath_feats

@torch.no_grad()
def test(model,loader,submetapath_feats,device):
    model.eval()
    raw_preds = []
    for batch, batch_feats, batch_labels_feats, batch_mask in loader:
        batch = batch.to(device)
        batch_feats = {k: x.to(device) for k, x in batch_feats.items()}
        batch_submetapath_feats = {k: x[batch].to(device) for k, x in submetapath_feats.items()}
        batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
        batch_mask = None
        if model.cfg["model"] == "SeHGNNver2": 
            pred = model(batch, batch_feats, batch_submetapath_feats, batch_labels_feats, batch_mask).cpu()
        else:
            pred = model(batch, batch_feats, batch_labels_feats, batch_mask).cpu()
        raw_preds.append(pred)
    raw_preds = torch.cat(raw_preds, dim=0).to(device)
    
    return raw_preds

def get_final_score(cfg,data,best_pred,full_loader,model,checkpt_file,labels,device):
    all_pred = torch.empty((data.num_nodes, cfg['nclass'])).to(device)
    all_pred[data.labeled_nid] = best_pred
    if len(full_loader):
        model.load_state_dict(torch.load(f'{checkpt_file}.pt', map_location='cpu'), strict=True)
        if not cfg["cpu"]: torch.cuda.empty_cache()
        submetapath_feats = model.submetapath_aggr(data.neighbor_aggr_feature_per_metapath) if model.cfg["model"] == "SeHGNNver2" else {}
        raw_preds = test(model,full_loader,submetapath_feats,device)
        all_pred[data.extra_nid] = raw_preds
    torch.save(all_pred, f'{checkpt_file}.pt')

    if cfg["dataset"] != 'IMDB':
        predict_prob = all_pred.softmax(dim=1)
    else:
        predict_prob = torch.sigmoid(all_pred)

    test_logits = predict_prob[data.test_nid]
    
    if cfg["dataset"] != 'IMDB':
        pred = test_logits.cpu().numpy().argmax(axis=1)
        data.dl.gen_file_for_evaluate(test_idx=data.test_nid, label=pred,true_label=labels[data.test_nid],file_name=f"{cfg.dataset}_{cfg.seeds}_{checkpt_file.split('/')[-1]}.txt")
    else:
        pred = (test_logits.cpu().numpy()>0.5).astype(int)
        data.dl.gen_file_for_evaluate(test_idx=data.test_nid, label=pred,true_label=labels[data.test_nid],file_name=f"{cfg.dataset}_{cfg.seeds}_{checkpt_file.split('/')[-1]}.txt", mode='multi')

    if cfg["dataset"] != 'IMDB':
        preds = predict_prob.argmax(dim=1, keepdim=True)
    else:
        preds = (predict_prob > 0.5).int()
    train_acc = evaluator(preds[data.train_nid], labels[data.train_nid])
    val_acc = evaluator(preds[data.val_nid], labels[data.val_nid])
    test_acc = evaluator(preds[data.test_nid], labels[data.test_nid])

    print(f'train_acc ({train_acc[0]*100:.2f}, {train_acc[1]*100:.2f}) ' \
        + f'val_acc ({val_acc[0]*100:.2f}, {val_acc[1]*100:.2f}) ' \
        + f'test_acc ({test_acc[0]*100:.2f}, {test_acc[1]*100:.2f})')
    print(checkpt_file.split('/')[-1])
    
    del model
    if device==f'cuda:{cfg.gpu_id}': torch.cuda.empty_cache()
    
    return train_acc,val_acc,test_acc


def run(cfg,model, data,optimizer,loader,device,scalar=None):
    train_times = []
    train_loader,eval_loader,full_loader = loader
    labels = data.labels 
    if cfg["dataset"] != 'IMDB':
        labels = labels.to(device)
        labels_cuda = labels.long().to(device)
    else: # dataset == (DBLP or ACM or Freebase)
        labels = labels.float().to(device)
        labels_cuda = labels.to(device)
    
    early_stopping = EarlyStopping(dataset=cfg['dataset'],patience=cfg['patience'],path=cfg['path'])
    loss_fcn = nn.BCEWithLogitsLoss() if cfg["dataset"] == 'IMDB' else nn.CrossEntropyLoss()
    train_loss_records,train_time_records,train_epoch_records = [],[],[]
        
    for epoch in tqdm(range(cfg['epochs'])):
        gc.collect()
        if device == f'cuda:{cfg.gpu_id}': torch.cuda.synchronize()
        start = time.time()
        loss, acc, submetapath_feats = train(model, data.feats, data.label_feats,data.neighbor_aggr_feature_per_metapath,labels_cuda, loss_fcn, optimizer, train_loader, evaluator,scalar=scalar)
        if device == f'cuda:{cfg.gpu_id}': torch.cuda.synchronize()
        end = time.time()
        
        log = f'Epoch {epoch}, training Time(s): {end-start:.4f}, estimated train loss {loss:.4f}, acc {acc[0]*100:.4f}, {acc[1]*100:.4f}\n'
        train_epoch_records.append(epoch)
        train_loss_records.append(loss)
        train_time_records.append(end-start) if epoch == 0 else train_time_records.append(train_time_records[epoch-1] +(end - start) )
        if device == f'cuda:{cfg.gpu_id}': torch.cuda.empty_cache()
        train_times.append(end-start)

        start = time.time()
        raw_preds = test(model,eval_loader,submetapath_feats,device)
        loss_train = loss_fcn(raw_preds[:data.trainval_point], labels[data.train_nid]).item()
        val_loss = loss_fcn(raw_preds[data.trainval_point:data.valtest_point], labels[data.val_nid]).item()
        test_loss = loss_fcn(raw_preds[data.valtest_point:data.labeled_num_nodes], labels[data.test_nid]).item()
        
        if cfg["dataset"] != 'IMDB':
            preds = raw_preds.argmax(dim=-1)
        else:
            preds = (raw_preds > 0.).int()

        train_acc = evaluator(preds[:data.trainval_point], labels[data.train_nid])
        val_acc = evaluator(preds[data.trainval_point:data.valtest_point], labels[data.val_nid])
        test_acc = evaluator(preds[data.valtest_point:data.labeled_num_nodes], labels[data.test_nid])

        end = time.time()
        log += f'evaluation Time: {end-start:.4f}, Train loss: {loss_train:.4f}, Val loss: {val_loss:.4f}, Test loss: {test_loss:.4f}\n'
        log += f'Train acc: ({train_acc[0]*100:.4f}, {train_acc[1]*100:.4f}), Val acc: ({val_acc[0]*100:.4f}, {val_acc[1]*100:.4f})'
        log += f', Test acc: ({test_acc[0]*100:.4f}, {test_acc[1]*100:.4f})\n'
        
        if early_stopping(model,epoch,log,val_loss,test_loss,val_acc,test_acc,raw_preds) is True:
            break
    
    best_pred = early_stopping.best_raw_preds
    checkpt_file = early_stopping.path[:-3]    
    print('average train times', sum(train_times) / len(train_times))
    early_stopping.print_best_epoch_result()
    # print(f'Best Epoch {best_epoch} at {checkpt_file}\n\tFinal Val loss {best_val_loss:.4f} ({best_val[0]*100:.4f}, {best_val[1]*100:.4f})'
    #         + f', Test loss {best_test_loss:.4f} ({best_test[0]*100:.4f}, {best_test[1]*100:.4f})')
    
    train_acc,val_acc,test_acc = get_final_score(cfg,data,best_pred,full_loader,model,checkpt_file,labels,device)
     
    return test_acc


@hydra.main(config_path='../conf', config_name='config')
def main(cfg):

    print(utils.get_original_cwd())
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment(cfg.experiment_name)
    mlflow.start_run()
    
    cfg = cfg[cfg.key]
    for key,value in cfg.items():
        mlflow.log_param(key,value)
        
    root = utils.get_original_cwd() + '/data/datasets/' + cfg['dataset']
    device = torch.device(f'cuda:{cfg.gpu_id}' if torch.cuda.is_available() else 'cpu')
    data = HeteroDataSet(cfg=cfg,root=root)
    preprocessing = PreProcessing(cfg=cfg).to(device)
    data  = preprocessing(data,model_name=cfg["model"])
    scalar = torch.cuda.amp.GradScaler()  if cfg['amp'] and device !='cpu' else None
    
    artifacts,test_accs_micro,test_accs_macro = {},[],[]
    for i in tqdm(range(cfg['run'])):
        print('Restart with seed =', i+1)
        set_random_seed(seed=i+1)
        set_checkpt_folder(cfg)
        data.get_training_setup()
        # clone raw_feats to avoid in-place modification for different seeds
        data.feats = {k: v.detach().clone() for k, v in data.raw_feats.items()}
        if cfg['label_feats']:
            data = preprocessing.compute_label_features(data)
            
        train_loader = torch.utils.data.DataLoader(data.train_nid, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)
        eval_loader, full_loader = get_eval_and_full_loader(cfg=cfg,data=data)
        loader = [train_loader,eval_loader,full_loader]
        if device == f'cuda:{cfg.gpu_id}':  torch.cuda.empty_cache()
        gc.collect()
        model = return_model(cfg,data).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg["lr"],weight_decay=cfg['weight_decay'])
        test_acc = run(cfg,model,data,optimizer,loader,device,scalar=scalar)
        
        test_accs_micro.append(test_acc[0])
        test_accs_macro.append(test_acc[1])
        
    # acc_max_index = test_accs.index(max(test_accs))
    test_accs_micro = [i * 100 for i in test_accs_micro]
    test_accs_macro = [i * 100 for i in test_accs_macro]
    
    test_acc_micro_ave = sum(test_accs_micro)/len(test_accs_micro)
    test_acc_macro_ave = sum(test_accs_macro)/len(test_accs_macro)
    
    # epoch_ave = sum(epochs)/len(epochs)
    # mlflow.log_metric('epoch_mean',epoch_ave)

    #f1_score(micro)
    mlflow.log_metric('test_acc_micro_min',min(test_accs_micro))
    mlflow.log_metric('test_acc_micro_mean',test_acc_micro_ave)
    mlflow.log_metric('test_acc_micro_max',max(test_accs_micro))
    
    #f1_score(macro)
    mlflow.log_metric('test_acc_macro_min',min(test_accs_macro))
    mlflow.log_metric('test_acc_macro_mean',test_acc_macro_ave)
    mlflow.log_metric('test_acc_macro_max',max(test_accs_macro))
    mlflow.end_run()
    return test_acc_micro_ave

    
    

if __name__ == "__main__":
    now_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()