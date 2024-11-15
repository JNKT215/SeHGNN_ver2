import math,os,gc,datetime
import dgl.function as fn 
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor,remove_diag
from utils import check_acc,save_and_load_dict_pkl
import metapath as mp
from sparse_tools import SparseAdjList

class PreProcessing(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self,data,model_name):
        max_length = int(self.cfg['num_hop']) + 1 
    
        # ---compute neighbor aggregaiton(dgl) mainmetapath ---
        print(f'Current num hops = {self.cfg.num_hop} for feature propagation')
        prop_tic = datetime.datetime.now()
        data = self.compute_l_hop_features(data=data)
        prop_toc = datetime.datetime.now()
        print(f'Time used for feat prop {prop_toc - prop_tic}')
        gc.collect()
                
        if model_name == "SeHGNNver2":
            # compute submetapth semantic fusion tensor
            print("etype_graph_dict")
            etype_graph_dict = self.get_etype_graph_dict(data)

            max_length = int(self.cfg['submetapath_hops']) + 1 
            metapath_name = mp.enum_metapath_name(data.edge_type,data.next_type,max_length)
            metapath_list = mp.enum_longest_metapath_index(data.edge_type,data.next_type,max_length)
            
            print("metapath_instance_dict_per_node")    
            metapath_instance_dict_per_node = {}
            for index in tqdm(range(data.total_nodes)):    
                tmp = mp.search_all_path(etype_graph_dict, index, metapath_name, metapath_list, data.edge_type,self.cfg["sampling_limit"])          
                metapath_instance_dict_per_node[index] = tmp   
                        
            print("neighbor_aggr_feature_per_metapath")
            neighbor_aggr_feature_per_metapath = self.calc_submetapath_neighbor_aggr_feature(data,metapath_instance_dict_per_node,echo=False)
            
            data.neighbor_aggr_feature_per_metapath = neighbor_aggr_feature_per_metapath
            
        return data
    
    def get_etype_graph_dict(self,data):
        etype_graph_dict = {}
        for e_type_index,e_type in data.edge_type.items():
            src,dst = e_type[0],e_type[1]
            src_node_type_index,dst_node_type_index = [i for i in range(data.node_slices[src][0],data.node_slices[src][1])],[i for i in range(data.node_slices[dst][0],data.node_slices[dst][1])]
        
            # e_type_edge_index = data.heterograph[e_type]['edge_index']
            e_type_edge_index = torch.stack([data.adjs[e_type].storage.row(),data.adjs[e_type].storage.col()])
            homograph_edge_index = e_type_edge_index.detach().clone()
        
            for col in range(e_type_edge_index.shape[1]):
                homograph_edge_index[0][col] = src_node_type_index[e_type_edge_index[0][col]] 
                homograph_edge_index[1][col] = dst_node_type_index[e_type_edge_index[1][col]] 
            
            print(e_type)
            print(homograph_edge_index)
        
            tmp = {i:[] for i in range(data.total_nodes)}
        
            for i in range(e_type_edge_index.shape[-1]):
                index = homograph_edge_index[0][i].item()
                tmp[index].append(homograph_edge_index[1][e_type_edge_index[1][i].item()].item()) 
            
            # etype_graph_dict[e_type] = tmp
            etype_graph_dict[e_type_index] = tmp        
        return etype_graph_dict
        
    def calc_submetapath_neighbor_aggr_feature(self,data,metapath_instance_dict_per_node,echo=False):
            def concatenate_features(x,metapath_key,homo_to_hetero_index_dict,indices):
                features = []
                for cnt,index in enumerate(indices):
                    node_type = metapath_key[cnt]
                    feature = x[node_type][homo_to_hetero_index_dict[index]]
                    features.append(feature)
                return torch.cat(features)
                  
            homo_to_hetero_index_dict,index = {},0
            for node_type in data.node_dict.keys():
                for cnt,_ in  enumerate(range(data.node_slices[node_type][0],data.node_slices[node_type][1])):
                    homo_to_hetero_index_dict[index] = cnt
                    index+=1
                    
            # すべてのノードとメタパスインスタンスを処理
            neighbor_aggr_feature_per_metapath = {}
            for node_id, metapaths in tqdm(metapath_instance_dict_per_node.items()):
                neighbor_aggr_feature_per_metapath[node_id] = {}
                for metapath_key, indices_list in metapaths.items():
                    #------（Neighbor Aggregation）----
                    concatenated_features = [concatenate_features(data.ntype_features,metapath_key,homo_to_hetero_index_dict,indices) for indices in indices_list]
                    concatenated_features = torch.stack(concatenated_features)         
                    if self.cfg["neighbor_encoder"] == "mean":
                        calc_metapath_insntance_feature_per_metapath = torch.mean(concatenated_features,dim=0)
                        neighbor_aggr_feature_per_metapath[node_id][metapath_key] = calc_metapath_insntance_feature_per_metapath
                    else: # neighbor_encoder == "sum"
                        neighbor_aggr_feature_per_metapath[node_id][metapath_key] = concatenated_features

            
            # # 結果を表示（neighbor_aggr_feature_per_metapath）
            if echo:
                for node_id, metapaths in neighbor_aggr_feature_per_metapath.items():
                    print(f"Node ID: {node_id}")
                    for metapath_key, features in metapaths.items():
                        print(f"  Metapath: {metapath_key}")
                        print(f"    Features: {features}")
            return neighbor_aggr_feature_per_metapath
    
    def hg_propagate_feat_dgl(self, g, tgt_type, num_hops, max_length, extra_metapath, echo=False):
        for hop in range(1, max_length):
            reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]
            for etype in g.etypes:
                stype, _, dtype = g.to_canonical_etype(etype)
                # if hop == args.num_hops and dtype != tgt_type: continue
                for k in list(g.nodes[stype].data.keys()):
                    if len(k) == hop:
                        current_dst_name = f'{dtype}{k}'
                        if (hop == num_hops and dtype != tgt_type and k not in reserve_heads) \
                        or (hop > num_hops and k not in reserve_heads):
                            continue
                        if echo: print(k, etype, current_dst_name)
                        g[etype].update_all(
                            fn.copy_u(k, 'm'),
                            fn.mean('m', current_dst_name), etype=etype)

            # remove no-use items
            for ntype in g.ntypes:
                if ntype == tgt_type: continue
                removes = []
                for k in g.nodes[ntype].data.keys():
                    if len(k) <= hop:
                        removes.append(k)
                for k in removes:
                    g.nodes[ntype].data.pop(k)
                if echo and len(removes): print('remove', removes)
            gc.collect()

            if echo: print(f'-- hop={hop} ---')
            for ntype in g.ntypes:
                for k, v in g.nodes[ntype].data.items():
                    print(f'{ntype} {k} {v.shape}', v[:,-1].max(), v[:,-1].mean())
            if echo: print(f'------\n')
        return g


    def hg_propagate_sparse_pyg(self, adjs, tgt_types, num_hops, max_length, extra_metapath, prop_feats=False, echo=False, prop_device='cpu'):
        store_device = 'cpu'
        if type(tgt_types) is not list:
            tgt_types = [tgt_types]

        label_feats = {k: v.clone() for k, v in adjs.items() if prop_feats or k[-1] in tgt_types} # metapath should start with target type in label propagation
        adjs_g = {k: v.to(prop_device) for k, v in adjs.items()}

        for hop in range(2, max_length):
            reserve_heads = [ele[-(hop+1):] for ele in extra_metapath if len(ele) > hop]
            new_adjs = {}
            for rtype_r, adj_r in label_feats.items():
                metapath_types = list(rtype_r)
                if len(metapath_types) == hop:
                    dtype_r, stype_r = metapath_types[0], metapath_types[-1]
                    for rtype_l, adj_l in adjs_g.items():
                        dtype_l, stype_l = rtype_l
                        if stype_l == dtype_r:
                            name = f'{dtype_l}{rtype_r}'
                            if (hop == num_hops and dtype_l not in tgt_types and name not in reserve_heads) \
                            or (hop > num_hops and name not in reserve_heads):
                                continue
                            if name not in new_adjs:
                                if echo: print('Generating ...', name)
                                if prop_device == 'cpu':
                                    new_adjs[name] = adj_l.matmul(adj_r)
                                else:
                                    with torch.no_grad():
                                        new_adjs[name] = adj_l.matmul(adj_r.to(prop_device)).to(store_device)
                            else:
                                if echo: print(f'Warning: {name} already exists')
            label_feats.update(new_adjs)

            removes = []
            for k in label_feats.keys():
                metapath_types = list(k)
                if metapath_types[0] in tgt_types: continue  # metapath should end with target type in label propagation
                if len(metapath_types) <= hop:
                    removes.append(k)
            for k in removes:
                label_feats.pop(k)
            if echo and len(removes): print('remove', removes)
            del new_adjs
            gc.collect()

        if prop_device != 'cpu':
            del adjs_g
            torch.cuda.empty_cache()

        return label_feats
    
    
    def compute_l_hop_features(self,data):
        if self.cfg['dataset'] in ['DBLP', 'ACM', 'IMDB']: #dataset == (DBLP, IMDB, ACM)
            if len(data.extra_metapath):
                max_length = max(self.cfg['num_hop'] + 1, max([len(ele) for ele in data.extra_metapath]))
            else:
                max_length = self.cfg['num_hop'] + 1

            ### gが常に初期の状態となっているかどうかを確認##
            g = self.hg_propagate_feat_dgl(data.g, data.tgt_type, self.cfg['num_hop'], max_length, data.extra_metapath, echo=True)
            
            raw_feats = {}
            keys = list(g.nodes[data.tgt_type].data.keys())

            for k in keys:
                raw_feats[k] = g.nodes[data.tgt_type].data.pop(k)
                
            print(f'For tgt type {data.tgt_type}, feature keys (num={len(raw_feats)}):', end='')
            
            print()
            for k, v in raw_feats.items():
                print(k, v.size())
            print()
            
            data.data_size = {k: v.size(-1) for k, v in raw_feats.items()}    
            data.feat_keys = keys
            data.raw_feats = raw_feats
            return data
        elif self.cfg['dataset'] == 'Freebase':
            if len(data.extra_metapath):
                max_length = max(self.cfg['num_hop'] + 1, max([len(ele) for ele in data.extra_metapath]))
            else:
                max_length = self.cfg['num_hop'] + 1

            if self.cfg['num_hop'] == 1:
                raw_meta_adjs = {k: v.clone() for k, v in data.adjs.items() if k[0] == data.tgt_type}
            else:
                save_name = f'./Freebase_adjs/feat_hop{self.cfg.num_hops}'
                if os.path.exists(f'{save_name}_00_int64.npy'):
                    raw_meta_adjs = {}
                    for srcname in data.dl.nodes['count'].keys():
                        print(f'Loading feature adjs from {save_name}_0{srcname}')
                        tmp = SparseAdjList(f'{save_name}_0{srcname}', None, None, data.num_tgt_nodes, data.dl.nodes['count'][srcname], with_values=True)
                        for k in tmp.keys:
                            assert k not in raw_meta_adjs
                        raw_meta_adjs.update(tmp.load_adjs(expand=True))
                        del tmp
                else:
                    print('Generating feature adjs for Freebase ...\n(For each configutaion, this happens only once for the first execution, and results will be saved for later executions)')
                    raw_meta_adjs = self.hg_propagate_sparse_pyg(data.adjs, data.tgt_type, self.cfg['num_hop'], max_length, data.extra_metapath, prop_feats=True, echo=True, prop_device=data.prop_device)

                    meta_adj_list = []
                    for srcname in data.dl.nodes['count'].keys():
                        keys = [k for k in raw_meta_adjs.keys() if k[-1] == str(srcname)]
                        print(f'Saving feature adjs {keys} into {save_name}_0{srcname}')
                        tmp = SparseAdjList(f'{save_name}_0{srcname}', keys, raw_meta_adjs, data.num_tgt_nodes, data.dl.nodes['count'][srcname], with_values=True)
                        meta_adj_list.append(tmp)

                    for srcname in data.dl.nodes['count'].keys():
                        tmp = SparseAdjList(f'{save_name}_0{srcname}', None, None, data.num_tgt_nodes, data.dl.nodes['count'][srcname], with_values=True)
                        tmp_adjs = tmp.load_adjs(expand=True)
                        print(srcname, tmp.keys)
                        for k in tmp.keys:
                            assert torch.all(raw_meta_adjs[k].storage.rowptr() == tmp_adjs[k].storage.rowptr())
                            assert torch.all(raw_meta_adjs[k].storage.col() == tmp_adjs[k].storage.col())
                            assert torch.all(raw_meta_adjs[k].storage.value() == tmp_adjs[k].storage.value())
                        del tmp_adjs, tmp
                        gc.collect()
                        

            raw_feats = {k: v.clone() for k, v in raw_meta_adjs.items() if len(k) <= self.cfg['num_hop'] + 1 or k in data.extra_metapath}

            assert '0' not in raw_feats
            raw_feats['0'] = SparseTensor.eye(data.dl.nodes['count'][0])
            
            print(f'For tgt type {data.tgt_type}, feature keys (num={len(raw_feats)}):', end='')
            
            print(' (in SparseTensor mode)')
            for k, v in raw_feats.items():
                print(k, v.sizes())
            
            data.data_size = dict(data.dl.nodes['count'])
            data.feat_keys = keys
            data.raw_feats = raw_feats
            data.raw_meta_adjs = raw_meta_adjs
            return data
        else:
            assert 0
    

    def compute_label_features(self,data):
        label_feats = {}
        if self.cfg["dataset"] != 'IMDB':
            label_onehot = torch.zeros((data.num_nodes, data.num_classes))
            label_onehot[data.train_nid] = F.one_hot(data.init_labels[data.train_nid], data.num_classes).float()
        else: # dataset == (DBLP or  ACM or Freebase)
            label_onehot = torch.zeros((data.num_nodes, data.num_classes))
            label_onehot[data.train_nid] = data.init_labels[data.train_nid].float()

        if self.cfg["dataset"] == 'DBLP':
            extra_metapath = []
        elif self.cfg["dataset"] == 'IMDB':
            extra_metapath = []
        elif self.cfg["dataset"] == 'ACM':
            extra_metapath = []
        elif self.cfg["dataset"] == 'Freebase':
            extra_metapath = []
        else:
            assert 0

        extra_metapath = [ele for ele in extra_metapath if len(ele) > self.cfg.num_label_hops + 1]
        if len(extra_metapath):
            max_length = max(self.cfg.num_label_hops + 1, max([len(ele) for ele in extra_metapath]))
        else:
            max_length = self.cfg.num_label_hops + 1

        print(f'Current num hops = {self.cfg.num_label_hops} for label propagation')
        # compute k-hop feature
        prop_tic = datetime.datetime.now()
        
        if self.cfg['dataset'] in  ['DBLP', 'ACM', 'IMDB']:
            meta_adjs = self.hg_propagate_sparse_pyg(
                        data.adjs, data.tgt_type, self.cfg.num_label_hops, max_length, extra_metapath, prop_feats=False, echo=True, prop_device=data.prop_device)
            
            print(f'For label propagation, meta_adjs: (in SparseTensor mode)')
            for k, v in meta_adjs.items():
                print(k, v.sizes())
            print()
            
            for k, v in tqdm(meta_adjs.items()):
                label_feats[k] = remove_diag(v) @ label_onehot
            gc.collect()

            if self.cfg["dataset"] == 'IMDB':
                condition = lambda ra,rb,rc,k: True
                check_acc(label_feats, condition, data.init_labels, data.train_nid, data.val_nid, data.test_nid, show_test=False, loss_type='bce')
            else: # dataset == (DBLP or ACM)
                condition = lambda ra,rb,rc,k: True
                check_acc(label_feats, condition, data.init_labels, data.train_nid, data.val_nid, data.test_nid, show_test=True)

        elif self.cfg['dataset'] ==  'Freebase':
            if self.cfg.num_label_hops <= self.cfg.num_hops and len(extra_metapath) == 0:
                meta_adjs = {k: v for k, v in data.raw_meta_adjs.items() if k[-1] == '0' and len(k) < max_length}        
            else:
                save_name = f'./Freebase_adjs/label_seed{self.cfg.seed}_hop{self.cfg.num_label_hops}'
                if self.cfg.seed > 0 and os.path.exists(f'{save_name}_int64.npy'):
                    meta_adj_list = SparseAdjList(save_name, None, None, data.num_tgt_nodes, data.num_tgt_nodes, with_values=True)
                    meta_adjs = meta_adj_list.load_adjs(expand=True)
                else:
                    meta_adjs = self.hg_propagate_sparse_pyg(
                        data.adjs, data.tgt_type, self.cfg.num_label_hops, max_length, extra_metapath, prop_feats=False, echo=True, prop_device=data.prop_device)
                    meta_adj_list = SparseAdjList(save_name, meta_adjs.keys(), meta_adjs, data.num_tgt_nodes, data.num_tgt_nodes, with_values=True)

                    tmp = SparseAdjList(save_name, None, None, data.num_tgt_nodes, data.num_tgt_nodes, with_values=True)
                    tmp_adjs = tmp.load_adjs(expand=True)
                    for k in tmp.keys:
                        assert torch.all(meta_adjs[k].storage.rowptr() == tmp_adjs[k].storage.rowptr())
                        assert torch.all(meta_adjs[k].storage.col() == tmp_adjs[k].storage.col())
                        assert torch.all(meta_adjs[k].storage.value() == tmp_adjs[k].storage.value())
                    del tmp_adjs, tmp
                    gc.collect()
            
            print(f'For label propagation, meta_adjs: (in SparseTensor mode)')
            for k, v in meta_adjs.items():
                print(k, v.sizes())
            print()

            if False:
                label_onehot_g = label_onehot.to(prop_device)
                for k, v in tqdm(meta_adjs.items()):
                    label_feats[k] = (remove_diag(v).to(prop_device) @ label_onehot_g).to(store_device)

                del label_onehot_g
                if not self.cfg.cpu: torch.cuda.empty_cache()
                gc.collect()

                condition = lambda ra,rb,rc,k: rb > 0.2
                check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=False)

                left_keys = ['00', '000', '0000', '0010', '0030', '0040', '0050', '0060', '0070']
                remove_keys = list(set(list(label_feats.keys())) - set(left_keys))
                for k in remove_keys:
                    label_feats.pop(k)
            else:
                left_keys = ['00', '000', '0000', '0010', '0030', '0040', '0050', '0060', '0070']
                remove_keys = list(set(list(meta_adjs.keys())) - set(left_keys))
                for k in remove_keys:
                    meta_adjs.pop(k)

                label_onehot_g = label_onehot.to(data.prop_device)
                for k, v in tqdm(meta_adjs.items()):
                    label_feats[k] = (remove_diag(v).to(data.prop_device) @ label_onehot_g).to(data.store_device)

                del label_onehot_g
                if not self.cfg.cpu: torch.cuda.empty_cache()
                gc.collect()


        print('Involved label keys', label_feats.keys())

        # label_feats = {k: v[init2sort] for k,v in label_feats.items()}
        prop_toc = datetime.datetime.now()
        print(f'Time used for label prop {prop_toc - prop_tic}')
        
        data.label_feats = label_feats
        data.label_feat_keys = label_feats.keys()
        return data

class SubMetapathAggr(nn.Module):
    def __init__(self,cfg,node_slices,hetero_g,ntype_feature,metapath_name):
        super().__init__()
        self.cfg = cfg
        self.node_slices = node_slices
        self.hetero_g = hetero_g
        self.ntypes = list(ntype_feature.keys())
        self.ntype_feature = ntype_feature
        self.ntype_num_node = {k:v.shape[0] for k,v in ntype_feature.items()}
        self.data_size = {k:v.shape[1] for k,v in ntype_feature.items()}
        self.metapath_name = metapath_name
        if self.cfg['sub_metapth_act'] == "leaky_relu":
            self.act = torch.nn.LeakyReLU(0.2)
        elif self.cfg['sub_metapth_act'] == "none":
            self.act = lambda x: x
        self.n_lin = nn.Linear(cfg["embed_size"],cfg["embed_size"])
        self.sub_metapath_atten_vector = nn.ParameterDict({})
        self.submetapath_input_drop = nn.Dropout(cfg["input_drop"])
        self.submetapath_embeding = nn.ParameterDict({})
        for ntype,metapaths in metapath_name.items():
            self.submetapath_embeding[ntype] =  nn.Parameter(torch.Tensor(self.ntype_feature[ntype].shape[-1], cfg["embed_size"]))
            for metapath in metapaths:
                input_dim = [self.ntype_feature[i].shape[-1] for i in metapath]
                self.submetapath_embeding[metapath] =  nn.Parameter(torch.Tensor(sum(input_dim) , cfg["embed_size"]))
            self.sub_metapath_atten_vector[ntype] = nn.Parameter(torch.empty(1,cfg["embed_size"]))
        self.reset_parameters()
    def reset_parameters(self):
        for k, v in self._modules.items():
            if isinstance(v, nn.ParameterDict):
                for _k, _v in v.items():
                    _v.data.uniform_(-0.5, 0.5)
            elif isinstance(v, nn.ModuleList):
                for block in v:
                    if isinstance(block, nn.Sequential):
                        for layer in block:
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                    elif hasattr(block, 'reset_parameters'):
                        block.reset_parameters()
            elif isinstance(v, nn.Sequential):
                for layer in v:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            elif hasattr(v, 'reset_parameters'):
                v.reset_parameters()
        self.n_lin.reset_parameters()
                

    def forward(self,neighbor_aggr_feature_per_metapath):
        semantic_fusion_per_nodetype_feature = self.calc_submetapath_semantic_fusion_feature(neighbor_aggr_feature_per_metapath=neighbor_aggr_feature_per_metapath,node_slices=self.node_slices,echo=False)
        
        for node_type in self.ntypes:
            self.hetero_g.nodes[node_type].data[node_type] = semantic_fusion_per_nodetype_feature[node_type].clone().to("cpu")
            gc.collect()
            
        self.hetero_g = self.hg_propagate_feat_dgl(g=self.hetero_g,num_hops=int(self.cfg['num_hop']),max_length=int(self.cfg['num_hop'])+1,echo=False,tgt_type_metapath_aggr=True)
        submetapath_feature_dict = {}
        feat_keys = list(self.hetero_g.nodes[self.cfg['tgt_type']].data.keys())
        
        for k in feat_keys:                
            submetapath_feature_dict[k] = self.hetero_g.nodes[self.cfg['tgt_type']].data.pop(k).to(f'cuda:{self.cfg.gpu_id}')
        gc.collect()
        
        for ntype,ntype_feature in self.ntype_feature.items():
            self.hetero_g.nodes[ntype].data[ntype] = ntype_feature.clone()


        return submetapath_feature_dict

    def hg_propagate_feat_dgl(self,g,num_hops, max_length, extra_metapath=[], echo=False,tgt_type_metapath_aggr=True):
        for hop in range(1, max_length):
            reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]
            for etype in g.etypes:
                stype, _, dtype = g.to_canonical_etype(etype)
                # if hop == args.num_hops and dtype != tgt_type: continue
                for k in list(g.nodes[stype].data.keys()):
                    if len(k) == hop:
                        current_dst_name = f'{dtype}{k}'
                        if tgt_type_metapath_aggr:
                            if (hop == num_hops and dtype != self.cfg["tgt_type"] and k not in reserve_heads) \
                            or (hop > num_hops and k not in reserve_heads):
                                continue
                        else:
                            if (hop == num_hops and dtype == self.cfg["tgt_type"] and k not in reserve_heads) \
                            or (hop > num_hops and k not in reserve_heads):
                                continue
                            
                        if echo: print(k, etype, current_dst_name)
                        g[etype].update_all(
                            fn.copy_u(k, 'm'),
                            fn.mean('m', current_dst_name), etype=etype)

            # remove no-use items
            for ntype in g.ntypes:
                if tgt_type_metapath_aggr:
                    if ntype == self.cfg['tgt_type']: continue
                else:
                    if ntype != self.cfg['tgt_type']: continue
                removes = []
                for k in g.nodes[ntype].data.keys():
                    if len(k) <= hop:
                        removes.append(k)
                for k in removes:
                    g.nodes[ntype].data.pop(k)
                if echo and len(removes): print('remove', removes)
            gc.collect()

            if echo: print(f'-- hop={hop} ---')
            for ntype in g.ntypes:
                for k, v in g.nodes[ntype].data.items():
                     if echo: print(f'{ntype} {k} {v.shape}', v[:,-1].max(), v[:,-1].mean())
            if echo: print(f'------\n')
        return g

        #ノードタイプに対応する，metapath の semantics を fusion する．最終的な出力としては，sub_metapath による表現を持たせることとする
      
    def calc_submetapath_semantic_fusion_feature(self,neighbor_aggr_feature_per_metapath,node_slices,echo=False):
        #semantic fusion
        submetapath_emmbedding_feature = []
        for _, metapaths in tqdm(neighbor_aggr_feature_per_metapath.items()):
            if self.cfg["neighbor_encoder"] == "mean":
                features_per_node = [self.submetapath_input_drop(feature.to(f'cuda:{self.cfg.gpu_id}') @ self.submetapath_embeding[metapath]) for metapath,feature in metapaths.items()]
            else: #self.cfg["neighbor_encoder"] == "sum"
                features_per_node = [torch.sum(self.submetapath_input_drop(feature.to(f'cuda:{self.cfg.gpu_id}') @ self.submetapath_embeding[metapath]),dim=0) for metapath,feature in metapaths.items()]
            semantic_fusion_feature_list_tensor = torch.stack(features_per_node)
            #------（Semantic Fusion）---- 
            if self.cfg['calc_type'] == "attention":
                target_type = list(metapaths.keys())[0]
                attn_score = self.act((self.sub_metapath_atten_vector[target_type] * torch.tanh(semantic_fusion_feature_list_tensor)).sum(-1))
                attn = F.softmax(attn_score, dim=0)
                semantic_fusion_feature = torch.sum(attn.view(len(metapaths),  -1) * semantic_fusion_feature_list_tensor, dim=0)
            elif self.cfg['calc_type'] == "mean": # mean
                semantic_fusion_feature = torch.mean(semantic_fusion_feature_list_tensor,dim=0)
            elif self.cfg['calc_type'] == "linear": # mean
                semantic_fusion_feature = torch.mean(semantic_fusion_feature_list_tensor,dim=0)
                semantic_fusion_feature = self.n_lin(semantic_fusion_feature)
            submetapath_emmbedding_feature.append(semantic_fusion_feature)  
            #------（Semantic Fusion）----
          
        submetapath_emmbedding_feature = torch.stack(submetapath_emmbedding_feature)
        feature_dict = {k:submetapath_emmbedding_feature[v[0]:v[1],:] for k,v in node_slices.items()}
        # 出力する
        if echo:
            for key, value in feature_dict.items():
                print(f'{key}: {value}')
            
        return feature_dict

def xavier_uniform_(tensor, gain=1.):
    fan_in, fan_out = tensor.size()[-2:]
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return torch.nn.init._no_grad_uniform_(tensor, -a, a)


class Transformer(nn.Module):
    '''
        The transformer-based semantic fusion in SeHGNN.
    '''
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none'):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if hasattr(v, 'reset_parameters'):
                v.reset_parameters()
        nn.init.zeros_(self.gamma)

    def forward(self, x, mask=None):
        B, M, C = x.size() # batchsize, num_metapaths, channels
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))

        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1) # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h) # [B, H, M, -1]
        return o.permute(0,2,1,3).reshape((B, M, C)) + x


class LinearPerMetapath(nn.Module):
    '''
        Linear projection per metapath for feature projection in SeHGNN.
    '''
    def __init__(self, cin, cout, num_metapaths):
        super(LinearPerMetapath, self).__init__()
        self.cin = cin #隠れ層
        self.cout = cout #出力層
        self.num_metapaths = num_metapaths

        self.W = nn.Parameter(torch.randn(self.num_metapaths, self.cin, self.cout))
        self.bias = nn.Parameter(torch.zeros(self.num_metapaths, self.cout))

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias.unsqueeze(0)


unfold_nested_list = lambda x: sum(x, [])

class SeHGNNver2(nn.Module):
    '''
        The SeHGNN model.
    '''
    def __init__(self,cfg,data):
        super(SeHGNNver2, self).__init__()
        self.cfg = cfg
        self.dataset = cfg["dataset"]
        self.feat_keys = sorted(data.feat_keys)
        self.label_feat_keys = sorted(data.label_feat_keys)
        self.num_channels = num_channels = 2 * len(self.feat_keys) + len(self.label_feat_keys) if self.cfg['neighbor_aggr_mode'] == "use_submetapath_for_semantic_fusion" else len(self.feat_keys) + len(self.label_feat_keys)
        self.tgt_type = cfg["tgt_type"]
        self.residual = cfg["residual"]

        self.input_drop = nn.Dropout(cfg["input_drop"])
        self.submetapath_aggr = SubMetapathAggr(cfg=self.cfg,node_slices=data.node_slices,hetero_g=data.g,ntype_feature=data.ntype_features,metapath_name=mp.enum_metapath_name(name_dict=data.edge_type,type_dict=data.next_type,length=int(self.cfg['submetapath_hops'])+1))
        self.sub_metapath_alpha_q = nn.ParameterDict({})
        self.metapath_lins = nn.ModuleDict({})
        for k in self.feat_keys:        
            self.sub_metapath_alpha_q[k] = nn.Parameter(torch.empty(cfg["hidden"]))
            self.metapath_lins[k] = nn.Linear(cfg["embed_size"], cfg["hidden"])
        self.q = nn.Parameter(torch.empty(1, cfg["hidden"] * (len(self.feat_keys) + 1 )))
        # self.k_lin = nn.Linear(cfg["hidden"] * (len(self.feat_keys) + 1 ),cfg["hidden"] * (len(self.feat_keys) + 1 ))
        self.k_lin = nn.Linear(cfg["embed_size"], cfg["hidden"])
        
        self.data_size = data.data_size
        self.embeding = nn.ParameterDict({})
        for k, v in self.data_size.items():
            self.embeding[str(k)] = nn.Parameter(torch.Tensor(v, cfg["embed_size"])) #metapath ごとの特徴変換

        if len(self.label_feat_keys):
            self.labels_embeding = nn.ParameterDict({})
            for k in self.label_feat_keys:
                self.labels_embeding[k] = nn.Parameter(torch.Tensor(cfg["nclass"], cfg["embed_size"]))
        else:
            self.labels_embeding = {}

        self.feature_projection = nn.Sequential(
            *([LinearPerMetapath(cfg["embed_size"], cfg["hidden"], num_channels),
               nn.LayerNorm([num_channels, cfg["hidden"]]),
               nn.PReLU(),
               nn.Dropout(cfg["dropout"]),]
            + unfold_nested_list([[
               LinearPerMetapath(cfg["hidden"], cfg["hidden"], num_channels),
               nn.LayerNorm([num_channels, cfg["hidden"]]),
               nn.PReLU(),
               nn.Dropout(cfg["dropout"]),] for _ in range(cfg["n_fp_layers"] - 1)])
            )
        )

        self.semantic_fusion = Transformer(cfg["hidden"], num_heads=cfg["num_heads"], att_drop=cfg["att_drop"], act=cfg["act"])
        self.fc_after_concat = nn.Linear(num_channels * cfg["hidden"], cfg["hidden"])

        if self.residual:
            self.res_fc = nn.Linear(cfg["embed_size"], cfg["hidden"])

        if self.dataset not in ['IMDB', 'Freebase']:
            self.task_mlp = nn.Sequential(
                *([nn.PReLU(),
                   nn.Dropout(cfg["dropout"]),]
                + unfold_nested_list([[
                   nn.Linear(cfg["hidden"], cfg["hidden"]),
                   nn.BatchNorm1d(cfg["hidden"], affine=False),
                   nn.PReLU(),
                   nn.Dropout(cfg["dropout"]),] for _ in range(cfg["n_task_layers"] - 1)])
                + [nn.Linear(cfg["hidden"], cfg["nclass"]),
                   nn.BatchNorm1d(cfg["nclass"], affine=False, track_running_stats=False)]
                )
            )
        else:
            self.task_mlp = nn.ModuleList(
                [nn.Sequential(
                    nn.PReLU(),
                    nn.Dropout(cfg["dropout"]))]
                + [nn.Sequential(
                    nn.Linear(cfg["hidden"], cfg["hidden"]),
                    nn.BatchNorm1d(cfg["hidden"], affine=False),
                    nn.PReLU(),
                    nn.Dropout(cfg["dropout"])) for _ in range(cfg["n_task_layers"] - 1)]
                + [nn.Sequential(
                    nn.Linear(cfg["hidden"], cfg["nclass"]),
                    nn.LayerNorm(cfg["nclass"], elementwise_affine=False),
                    )]
            )
        
        
        
        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if isinstance(v, nn.ParameterDict):
                for _k, _v in v.items():
                    _v.data.uniform_(-0.5, 0.5)
            elif isinstance(v, nn.ModuleList):
                for block in v:
                    if isinstance(block, nn.Sequential):
                        for layer in block:
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                    elif hasattr(block, 'reset_parameters'):
                        block.reset_parameters()
            elif isinstance(v, nn.Sequential):
                for layer in v:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            elif hasattr(v, 'reset_parameters'):
                v.reset_parameters()
        self.k_lin.reset_parameters()
        self.q.data.uniform_(-0.5, 0.5)

    def forward(self,batch,feature_dict,submetapath_feature_dict, label_dict={}, mask=None):
        if isinstance(feature_dict[self.tgt_type], torch.Tensor):
            features = {k: self.input_drop(x @ self.embeding[k]) for k, x in feature_dict.items()}
        elif isinstance(feature_dict[self.tgt_type], SparseTensor):
            # Freebase has so many metapaths that we use feature projection per target node type instead of per metapath
            features = {k: self.input_drop(x @ self.embeding[k[-1]]) for k, x in feature_dict.items()}
        else:
            assert 0
            
        B = num_node = features[self.tgt_type].shape[0]
        C = self.num_channels #metapath
        D = features[self.tgt_type].shape[1]
        labels = {k: self.input_drop(x @ self.labels_embeding[k]) for k, x in label_dict.items()}
        
        if self.cfg['neighbor_aggr_mode'] == "only_submetapth_feature":
            x = [submetapath_feature_dict[k] for k in self.feat_keys] + [labels[k] for k in self.label_feat_keys]
        elif self.cfg['neighbor_aggr_mode'] == "all":
            alpha = self.cfg['submetapath_feature_weight']
            x = [ ((1-alpha) * features[k]) + (alpha * submetapath_feature_dict[k]) for k in self.feat_keys] + [labels[k] for k in self.label_feat_keys]
        elif self.cfg['neighbor_aggr_mode'] == "attention":
            alpha_dict = {key:self.sub_metapath_attention(features[key],submetapath_feature_dict[key],self.sub_metapath_alpha_q[key]) for key in self.feat_keys}
            x = [ (alpha_dict[k][0] * features[k]) + (alpha_dict[k][1] * submetapath_feature_dict[k]) for k in self.feat_keys] + [labels[k] for k in self.label_feat_keys]
        elif self.cfg['neighbor_aggr_mode'] == "all_ver3":
            sub_features = [submetapath_feature_dict[sub_key] for sub_key in self.feat_keys]
            alpha_dict = self.sub_metapath_attention_ver2(features,sub_features)
            x = [  features[k] + (alpha_dict[k] * submetapath_feature_dict[k]) for k in self.feat_keys] + [labels[k] for k in self.label_feat_keys]
        elif self.cfg['neighbor_aggr_mode'] == "use_submetapath_for_semantic_fusion":
            x =  [features[k] for k in self.feat_keys] + [submetapath_feature_dict[k] for k in self.feat_keys] + [labels[k] for k in self.label_feat_keys]
 

        # x = [features[k] for k in self.feat_keys] + [labels[k] for k in self.label_feat_keys]
        x = torch.stack(x, dim=1) # [B, C, D]
        x = self.feature_projection(x)
        x = self.semantic_fusion(x, mask=None).transpose(1,2)
            
        x = self.fc_after_concat(x.reshape(B, -1))
        if self.residual:
            x = x + self.res_fc(features[self.tgt_type])

        if self.dataset not in ['IMDB', 'Freebase']:
            return self.task_mlp(x)
        else:
            x = self.task_mlp[0](x)
            for i in range(1, len(self.task_mlp)-1):
                x = self.task_mlp[i](x) + x
            x = self.task_mlp[-1](x)
            return x
        
    def sub_metapath_attention(self,main_feature,sub_feature,q):
        out = torch.stack([main_feature,sub_feature])
        attn_score = (q * torch.tanh(self.k_lin(out)).mean(1)).sum(-1)
        attn = F.softmax(attn_score, dim=0)
        main_alpha,sub_alpha = attn[0].item(),attn[1].item()
        return main_alpha,sub_alpha
    
    def sub_metapath_attention_ver2(self,features,sub_features):
        concat_list = [ torch.cat([features[k]] + sub_features, dim=1) for k in self.feat_keys]
        out = torch.stack(concat_list)
        attn_score = (self.q * torch.tanh(out).mean(1)).sum(-1)
        attn = F.softmax(attn_score, dim=0)
        # att_dict = {metapath:(1 - attn[index].item(),attn[index].item())for index,metapath in enumerate(self.feat_keys)}
        att_dict = {metapath:attn[index].item()for index,metapath in enumerate(self.feat_keys)}
        return att_dict

class SeHGNN(nn.Module):
    '''
        The SeHGNN model.
    '''
    def __init__(self,cfg,feat_keys,label_feat_keys,data_size=None):
        super(SeHGNN, self).__init__()
        self.cfg = cfg
        self.dataset = cfg["dataset"]
        self.feat_keys = sorted(feat_keys)
        self.label_feat_keys = sorted(label_feat_keys)
        self.num_channels = num_channels = len(self.feat_keys) + len(self.label_feat_keys)
        self.tgt_type = cfg["tgt_type"]
        self.residual = cfg["residual"]

        self.input_drop = nn.Dropout(cfg["input_drop"])

        self.data_size = data_size
        self.embeding = nn.ParameterDict({})
        for k, v in data_size.items():
            self.embeding[str(k)] = nn.Parameter(torch.Tensor(v, cfg["embed_size"])) #metapath ごとの特徴変換

        if len(self.label_feat_keys):
            self.labels_embeding = nn.ParameterDict({})
            for k in self.label_feat_keys:
                self.labels_embeding[k] = nn.Parameter(torch.Tensor(cfg["nclass"], cfg["embed_size"]))
        else:
            self.labels_embeding = {}

        self.feature_projection = nn.Sequential(
            *([LinearPerMetapath(cfg["embed_size"], cfg["hidden"], num_channels),
               nn.LayerNorm([num_channels, cfg["hidden"]]),
               nn.PReLU(),
               nn.Dropout(cfg["dropout"]),]
            + unfold_nested_list([[
               LinearPerMetapath(cfg["hidden"], cfg["hidden"], num_channels),
               nn.LayerNorm([num_channels, cfg["hidden"]]),
               nn.PReLU(),
               nn.Dropout(cfg["dropout"]),] for _ in range(cfg["n_fp_layers"] - 1)])
            )
        )

        self.semantic_fusion = Transformer(cfg["hidden"], num_heads=cfg["num_heads"], att_drop=cfg["att_drop"], act=cfg["act"])
        self.fc_after_concat = nn.Linear(num_channels * cfg["hidden"], cfg["hidden"])

        if self.residual:
            self.res_fc = nn.Linear(cfg["embed_size"], cfg["hidden"])

        if self.dataset not in ['IMDB', 'Freebase']:
            self.task_mlp = nn.Sequential(
                *([nn.PReLU(),
                   nn.Dropout(cfg["dropout"]),]
                + unfold_nested_list([[
                   nn.Linear(cfg["hidden"], cfg["hidden"]),
                   nn.BatchNorm1d(cfg["hidden"], affine=False),
                   nn.PReLU(),
                   nn.Dropout(cfg["dropout"]),] for _ in range(cfg["n_task_layers"] - 1)])
                + [nn.Linear(cfg["hidden"], cfg["nclass"]),
                   nn.BatchNorm1d(cfg["nclass"], affine=False, track_running_stats=False)]
                )
            )
        else:
            self.task_mlp = nn.ModuleList(
                [nn.Sequential(
                    nn.PReLU(),
                    nn.Dropout(cfg["dropout"]))]
                + [nn.Sequential(
                    nn.Linear(cfg["hidden"], cfg["hidden"]),
                    nn.BatchNorm1d(cfg["hidden"], affine=False),
                    nn.PReLU(),
                    nn.Dropout(cfg["dropout"])) for _ in range(cfg["n_task_layers"] - 1)]
                + [nn.Sequential(
                    nn.Linear(cfg["hidden"], cfg["nclass"]),
                    nn.LayerNorm(cfg["nclass"], elementwise_affine=False),
                    )]
            )
        
        
        
        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if isinstance(v, nn.ParameterDict):
                for _k, _v in v.items():
                    _v.data.uniform_(-0.5, 0.5)
            elif isinstance(v, nn.ModuleList):
                for block in v:
                    if isinstance(block, nn.Sequential):
                        for layer in block:
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                    elif hasattr(block, 'reset_parameters'):
                        block.reset_parameters()
            elif isinstance(v, nn.Sequential):
                for layer in v:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            elif hasattr(v, 'reset_parameters'):
                v.reset_parameters()

    def forward(self, batch, feature_dict, label_dict={}, mask=None):
        if isinstance(feature_dict[self.tgt_type], torch.Tensor):
            features = {k: self.input_drop(x @ self.embeding[k]) for k, x in feature_dict.items()}
        elif isinstance(feature_dict[self.tgt_type], SparseTensor):
            # Freebase has so many metapaths that we use feature projection per target node type instead of per metapath
            features = {k: self.input_drop(x @ self.embeding[k[-1]]) for k, x in feature_dict.items()}
        else:
            assert 0

        B = num_node = features[self.tgt_type].shape[0]
        C = self.num_channels #metapath
        D = features[self.tgt_type].shape[1]

        labels = {k: self.input_drop(x @ self.labels_embeding[k]) for k, x in label_dict.items()}

        x = [features[k] for k in self.feat_keys] + [labels[k] for k in self.label_feat_keys]
        x = torch.stack(x, dim=1) # [B, C, D]
        x = self.feature_projection(x)

        x = self.semantic_fusion(x, mask=None).transpose(1,2)

        x = self.fc_after_concat(x.reshape(B, -1))
        if self.residual:
            x = x + self.res_fc(features[self.tgt_type])

        if self.dataset not in ['IMDB', 'Freebase']:
            return self.task_mlp(x)
        else:
            x = self.task_mlp[0](x)
            for i in range(1, len(self.task_mlp)-1):
                x = self.task_mlp[i](x) + x
            x = self.task_mlp[-1](x)
            return x



def return_model(cfg,data):
    if cfg['model'] == 'SeHGNN':
        model = SeHGNN(cfg=cfg,feat_keys=data.feat_keys,label_feat_keys=data.label_feat_keys,data_size=data.data_size)
    if cfg['model'] == 'SeHGNNver2':
        model = SeHGNNver2(cfg=cfg,data=data)
    return model