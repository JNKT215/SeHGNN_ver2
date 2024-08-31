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

    def forward(self,data,model_name,commmon_path):
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
            
            print("homo_to_hetero_index_dict")
            homo_to_hetero_index_dict = self.get_homo_to_hetero_index_dict(data)
            
            print("neighbor_aggr_feature_per_metapath")
            neighbor_aggr_feature_per_metapath = self.calc_submetapath_neighbor_aggr_feature(data,homo_to_hetero_index_dict,metapath_instance_dict_per_node,echo=False)
            
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
    
    def get_homo_to_hetero_index_dict(self,data):
        homo_to_hetero_index_dict = {}
        index = 0
        for node_type,node_type_index in tqdm(data.node_dict.items()):
            cnt = 0
            for _ in  range(data.node_slices[node_type][0],data.node_slices[node_type][1]):
                homo_to_hetero_index_dict[index] = cnt
                cnt+=1
                index+=1
        return homo_to_hetero_index_dict
    
    def calc_submetapath_neighbor_aggr_feature(self,data,homo_to_hetero_index_dict,metapath_instance_dict_per_node,echo=False):
            def concatenate_features(node_id,x,metapath_key, indices):
                features = []
                for cnt,index in enumerate(indices):
                    node_type = metapath_key[cnt]
                    feature = x[node_type][homo_to_hetero_index_dict[index]]
                    features.append(feature)
                return torch.cat(features)
            
            neighbor_aggr_feature_per_metapath = {}
            # すべてのノードとメタパスインスタンスを処理
            neighbor_aggr_feature_per_metapath = {}
            for node_id, metapaths in tqdm(metapath_instance_dict_per_node.items()):
                neighbor_aggr_feature_per_metapath[node_id] = {}
                for metapath_key, indices_list in metapaths.items():
                    concatenated_features_list = []
                    for indices in indices_list:
                        concatenated_features = concatenate_features(node_id,data.ntype_features,metapath_key, indices)
                        concatenated_features_list.append(concatenated_features)
                    #Neighbor_Aggr の計算部分にあたる
                    #------（Neighbor Aggregation）----
                    concatenated_features_list_tensor = torch.stack(concatenated_features_list)
                    calc_metapath_insntance_feature_per_metapath = torch.sum(concatenated_features_list_tensor,dim=0)
                    calc_metapath_insntance_feature_per_metapath = calc_metapath_insntance_feature_per_metapath / len(indices_list) 
                    #------（Neighbor Aggregation）----
                    neighbor_aggr_feature_per_metapath[node_id][metapath_key] = calc_metapath_insntance_feature_per_metapath

            
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

        self.embeding_concat_ver2 = nn.ModuleDict({})
        for ntype,metapaths in metapath_name.items():
            self.embeding_concat_ver2[ntype] =  nn.Sequential(
                                                                *([LinearPerSubMetapath(self.ntype_feature[ntype].shape[-1], cfg["embed_size"]),
                                                                nn.LayerNorm([cfg["embed_size"]]),
                                                                nn.PReLU(),
                                                                nn.Dropout(cfg["dropout"]),]
                                                                + unfold_nested_list([[
                                                                LinearPerSubMetapath(cfg["embed_size"], cfg["embed_size"]),
                                                                nn.LayerNorm([cfg["embed_size"]]),
                                                                nn.PReLU(),
                                                                nn.Dropout(cfg["dropout"]),] for _ in range(cfg["n_fp_submetapath_layers"] - 1)])
                                                                )
                                                            ) 
            for metapath in metapaths:
                input_dim = [self.ntype_feature[i].shape[-1] for i in metapath]
                self.embeding_concat_ver2[metapath] =  nn.Sequential(
                                                                        *([LinearPerSubMetapath(sum(input_dim), cfg["embed_size"]),
                                                                        nn.LayerNorm([cfg["embed_size"]]),
                                                                        nn.PReLU(),
                                                                        nn.Dropout(cfg["dropout"]),]
                                                                        + unfold_nested_list([[
                                                                        LinearPerSubMetapath(cfg["embed_size"], cfg["embed_size"]),
                                                                        nn.LayerNorm([cfg["embed_size"]]),
                                                                        nn.PReLU(),
                                                                        nn.Dropout(cfg["dropout"]),] for _ in range(cfg["n_fp_submetapath_layers"] - 1)])
                                                                        )
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
                
    def test_batch(self):
        if self.cfg.dataset == "DBLP":
            batch = torch.tensor([1584, 3697,  432, 2004, 2701, 1880, 1225, 1012,  872, 3723, 1800, 2072,
                                1318, 4004,  711, 3577, 2534, 2906, 1596,  428, 2138, 2365, 2661,  805,
                                1945, 3738, 3205, 2938, 1881, 2030, 1016, 2020, 2107, 3344, 3592,  469,
                                3932, 3138,  338,  833,  694, 3207,  699, 1108, 3270, 2438, 3443, 3626,
                                1096, 2150, 2137, 1194,  315, 1125, 2831, 2817, 2895, 3431, 3791, 3284,
                                3238, 3988, 1143, 1662, 1320,   47, 1620, 2068, 2634, 1780,  344, 2698,
                                1367, 2849, 3080,  374, 2756, 2389,  203,  659, 2169, 2940, 2226, 2618,
                                427, 3559, 1699, 1628, 2919,  930, 1985,  117, 2541, 2967,  370, 1559,
                                1395, 3022, 3672, 3400, 2304, 2680, 4050, 2451, 2931, 2128,  240, 1672,
                                2650, 1135, 1400, 2266, 3145, 1506, 1278, 1416,  438, 1786,  107,  103,
                                1990, 2515, 1742,  278, 1102, 1422, 3863,  397,  118, 2986, 3565, 3867,
                                1224, 1034,  181, 3547,  904, 1007,  295, 1570, 1533, 3989,  237, 1103,
                                447, 3896,  963,  975,  802,  979, 1860,  719,  959, 2021, 1861, 3504,
                                1040, 3162, 2385, 3133,  332, 3101, 1969,  759,  323, 1636, 3081, 3492,
                                3973, 1707, 3512, 3533, 2627, 2778, 3461, 2262, 1119, 1865,  578,  467,
                                1070, 3746, 1483, 2671,  489, 4002, 3802, 3203, 1835, 3437, 1654,  907,
                                2214, 1362, 1186, 3829, 1207,  928, 2620, 3475, 1273, 2802, 2094, 3386,
                                24, 1879, 1419, 1496, 4009, 2199, 2614, 1794, 3272, 3958, 2061,  361,
                                3812, 3423,  834, 3055, 1583, 3948,  293, 2537, 2743, 4032,  942, 2994,
                                2272,  667, 2579, 1964, 3712, 2587,  114, 1078, 2674, 2952, 1214,  623,
                                2225, 2463, 4045, 2045, 3522, 1858, 2098,  454, 3553,  371, 2269,  227,
                                2883,  795, 3177, 2364, 1039,  441, 1714, 3306, 2565, 3234, 3471,  517,
                                994,  458, 1783, 2513,  334,  714,  596, 1261, 1965, 2696, 3051, 2087,
                                847, 3885, 3151, 2194,  423, 2410, 1532,  131,  449, 2527, 1254, 1891,
                                647, 3155, 2987, 3776,  307, 2586, 2816,  294, 2767, 3750, 1187, 3610,
                                905, 1182,  177, 3163,  779,  624, 2866,  846, 3324, 3646, 2139, 1188,
                                1123,    7, 2510,  746, 1323, 1093,  246, 1445, 2550, 3961, 1137, 2957,
                                3071, 1343, 3109, 2833,  405, 2570, 1279,  725, 1575,    9,  120, 1304,
                                526, 2605, 1132, 4001,  259,  598,  832, 1494, 1730,  556, 2655,  903,
                                2810, 1210, 2714, 1238,  983, 1947,  799,  226, 3803, 1775, 3671, 3439,
                                3267, 2165,  985, 3120,  383,  202,  255, 3773, 2846, 1710, 2001, 1663,
                                2889, 3661,  197,  823, 2841, 1252, 2057, 2512,  997, 1542,  882,  354,
                                215, 1980,  140, 2557, 2155, 3282,  723, 3165, 3825, 1107,  507,  518,
                                589, 3501,  393, 2069, 3952, 1450, 3552, 2859, 3765,  372, 1919, 3097,
                                2182,  270, 3043, 1086, 2925, 1655, 3853,  537, 1466,  686,  464, 4037,
                                1553,  211, 3752, 3536, 3095, 1059,  324, 1443, 1949, 2840, 4011, 1694,
                                3554, 4030,  638, 2520, 1717, 2355,  739, 3342, 1741, 1349, 3949, 3359,
                                1762, 1456, 2836, 3196,   37, 2528, 2424, 1950, 3289,  950,  758, 1444,
                                2577, 1732, 1315, 3104,   15, 2669, 3922, 1484, 2126, 2551, 2972, 1013,
                                350,  971, 2117,  729, 1271,  230, 4031, 3601, 1248, 2929, 1573, 3413,
                                408, 3067, 1930, 2867, 1526,   55, 3575, 2690, 2921, 4018, 2170, 3645,
                                3321,  926,  862, 3497, 1952, 3026, 1925, 1683, 2011,  231, 3053,  656,
                                2532, 3696,  606, 3305, 3184, 2246,  748,  631,   66,  340, 2044,  184,
                                2976,  669, 3971,  775, 2658, 3371,  531, 2844, 3634,  431, 3263, 1047,
                                2035, 1927, 3468, 1421, 1973, 1726, 3603, 3008, 2036, 1375, 3016, 1910,
                                2969, 3628, 3410, 3758, 3294,  968, 1140,  466, 3642, 2051,  895, 2251,
                                2179, 2643,  603, 2466, 1931, 3721, 2837,  635, 3874, 3278, 1420, 1166,
                                535, 2649, 2592, 1517, 2989, 4021, 2760, 3943,  644, 3641, 1332,  703,
                                732, 1270, 3052,  653, 3729, 2491, 3467, 1246, 2544, 1995,  892,   19,
                                2209, 3419, 2275, 4006, 1820, 3995, 2588, 1227, 3600, 1999, 1330, 2677,
                                1230, 1687, 1460,  931, 3434, 2726, 2898,  745, 1259, 3534,  690,  475,
                                2302, 3018, 1480,  584,   75, 2768, 3195, 3046, 4036,  247,  155, 4029,
                                1705,  390, 1855, 1436, 1361, 1866, 2183, 1630, 3017, 3174, 2462, 2033,
                                3735, 1635, 1385, 2602, 1408, 3487, 3878,  888,   62, 1870, 1619,  506,
                                2405,   70,  938, 2492, 2796,    0, 2022, 4025,   63, 3171,  943, 3453,
                                2645, 3774,   36,  273, 4016, 3570, 1110, 1120, 2070, 1091, 2434, 2259,
                                3804,  562, 3091, 3621,  362,  813, 1205, 3140, 1364, 3814, 1502, 2901,
                                2472, 3198, 1338,  674,  933, 2237, 1352, 2186,   10, 4051,  844, 3290,
                                1847,  867, 2457,  539,  166,  544,  677, 2134, 2860, 2303, 2449, 2688,
                                1267, 1005,   68,  780, 3987,  314, 4013, 2490, 2476, 1534,  290, 2573,
                                2621, 2631,  224, 3096, 2881, 2517, 2715,  866, 3576, 2958,  853,  671,
                                3743, 3751, 1785, 3251, 1234, 3077,   94, 3900, 1761, 1977,  156, 1819,
                                1381, 3142,  608, 3851, 3981, 1796, 3092,  477,   71, 2374, 3703, 2343,
                                3530, 1288, 3639,  627,  782, 3334, 1474, 3784, 1709,  877, 3710, 2129,
                                1055, 3732, 1522, 4023, 2773, 3564, 2878, 2159, 3368,  724, 1704, 1825,
                                2049, 1430, 3568,  776, 4046, 1996,  434,  375, 3665,  590, 3720, 1072,
                                2523, 3718, 3906, 1900, 1325, 2153, 1231, 3303, 2162,  139, 1991, 2047,
                                1675, 1912, 1272, 3021, 1613,  885,  984, 3039, 3200,  743,  151,  591,
                                3231,  750, 1760,  756, 3782,  420, 3070, 2391, 2406, 1136, 1301, 3852,
                                3114, 3474, 1998,  260, 3873, 2578, 2642, 3291, 3998, 4017,  382, 2823,
                                1284, 3823, 3663, 3045, 3815, 1247,  767, 3631, 1065,  880, 2417, 2100,
                                2393, 2354, 3902, 1462, 2337, 3031,  923,  318, 3176,  891, 1491, 3193,
                                1454, 2740,  585,  322, 2111, 3309, 1258, 1918, 1253, 2597,  601, 3010,
                                4003, 2358, 1151, 1142, 2034, 2247, 2131, 1376, 2640,  634, 1548,  525,
                                3332,  783, 3833,  572, 2017, 3313,  967, 3214, 1828, 2553, 2535,  857,
                                91, 2735, 2616, 1638,  445,   98, 3212, 2827,  395,  766, 2567, 3944,
                                3202, 1097, 3481, 2741, 1528, 1802, 1660, 3099, 3042, 3312,  586, 3624,
                                1907, 2626,  452, 2185, 3541, 3388, 3956, 1637,   12, 2608, 1691, 2190,
                                2394, 2335,  141,  819, 3178, 1061, 1882, 3602, 1087,  214, 2447, 1467,
                                3343, 2381,  276, 3730, 2261, 2711, 3761,   76,  995, 1667, 1556, 2429,
                                1897,  744, 2850, 2279, 2801,  195, 1122, 2426,  523,  896, 1895, 3076,
                                2548, 2441])
        elif self.cfg.dataset == "ACM":
            batch = torch.tensor([1561, 1562, 2350,  658,  689, 1720, 2333,  681, 1407,  684, 1894,  356,
                                1417, 1576, 2504,  862, 1493, 2001, 2461, 1924, 1672, 2153, 1344, 1710,
                                2175,  161,  516,  496, 1372,  627,  696, 2279, 1974, 1234,   45, 2977,
                                337, 2635, 2064,  477,  755, 1035, 1701,  670, 2443, 2365, 1869, 1659,
                                2060,  722,  278,   62, 1329, 2370,  111, 2221,  296, 3016,  176,  646,
                                301,  698,  494, 1325, 2184,  880,  544,  355, 1611,  311, 1323,  521,
                                2962, 1556, 1984,  333, 1388, 2670, 1171, 1517, 2901, 1958, 1887, 1478,
                                2866, 3000, 1397, 1006,   25, 2556, 1596, 2478,  991,  824,  675, 1433,
                                1195, 1697,  820, 2987,  810,  135, 2259, 2532,  789, 2540, 1792,  200,
                                2371, 2535,  581, 2905, 2498,  241, 2868, 1866, 1591,   93,  119,  210,
                                2228, 1938,  560,  304,  275,  103, 1991,  901, 2830, 1078, 2844, 2151,
                                1265,  433, 1505, 2243, 2858,  270,  785, 2610, 1331,  216, 2137, 2687,
                                2903, 2992, 1636,  160, 1332, 2509, 2894,  284, 2523,  126,  346, 2862,
                                102,  155,  872, 1983, 2415, 1595, 2018, 1552, 2967, 2205, 1186,  701,
                                259, 2579, 2728, 3024,  534, 1766,  341,  891,  860, 2397, 1993, 2848,
                                731, 2738,  137, 1896, 1041,  979,  485,  258, 2677, 2696, 2794, 1832,
                                2080, 1730, 2035, 1150,   10, 2122, 1684, 2541, 1608, 2951, 1405,  371,
                                2150, 2573, 1076, 1396,  803, 1787, 2653,  652, 1172,  353, 2565, 2533,
                                1780,  555, 1873,   41, 2619,  999, 1225, 2999,  579, 1255, 1847, 1981,
                                1951,  295, 1616,  481, 2483,  813, 2286,  109,  630, 1221, 2487, 1381,
                                2994, 2112,  952, 1305, 1749, 1512, 1791, 1461, 2422, 1753, 1821,   90,
                                1111, 1448,  759,  417,  116,  447,  664,  941,  580,  214, 2578,  980,
                                47, 1484, 1631, 1897, 1648, 1588, 1114,   16,  386, 1363, 1880,   19,
                                2211,  854,  144, 1953, 2815, 2874, 2334,  881, 2059, 2473, 1960,  766,
                                2177,   80, 2680, 2731, 2146, 2217, 2027, 2303,  542, 2883, 2156, 1902,
                                588, 2196, 1226,  469,   28, 2198,   65, 2481, 2557,  626, 1700, 1132,
                                2503, 2100, 1423,  226,  281,  388,  504, 1161,  404,  372,  597,  776,
                                61, 2667, 1784, 2210, 1275, 1449, 2769,  310, 1485,  836, 1815,  308,
                                1246,  651,  717, 2800, 1304, 1586, 1311, 2054, 1593, 2229, 1403,  242,
                                172,  313,  506, 2733,  456, 1465, 2588, 2138,  693, 1703,  732, 2313,
                                1294, 2668, 2688,  196, 1770,  948, 2517, 2654, 1382, 2272, 1479, 2234,
                                2466, 1373, 2970, 2245, 2725, 2880,  139, 2394,  680,  982, 1462, 1259,
                                2212, 3003,  131, 2679,  764, 2038, 1572, 1439,  942,  156,  518, 2692,
                                2374,  771, 2840, 2320,  357,  746,    7,  206, 2877,  870, 2323,  705,
                                842, 2742, 1415, 1838,  331, 1389, 1946, 1156,  886,   75, 1336,  795,
                                1871, 1802, 2645, 2710, 1217, 1279,  522, 1769,  966,  921, 2180, 1501,
                                2149,  240,  563,  943,  455, 1589, 2820,  148,  193, 2850,  195, 2525,
                                1707, 2598, 1702, 1990, 1782, 1349,  541, 1103,  224,  228,  833, 2612,
                                1982, 1330,  934,  165, 2287,  554,  832,  229,  487,  344, 2754,  804,
                                467, 1973,  499, 2337,  462, 1251, 1044, 1804, 2661,  965, 1201,   36,
                                959,  743, 1756,  366, 1192, 1899,  725,  988, 2544,  598, 2966, 1466,
                                566, 2079, 1367, 2036,  419,  674,  735, 2193,    4,  614,  321, 1923,
                                2067, 1059, 1202, 2441,  528, 1967, 1640, 2016,  413, 1581,  265,  923,
                                1394, 2873, 2119, 1050, 2472,  949, 1719,   51,  882, 2496, 2950,   59,
                                753,  586,  114, 2824,  903, 2142,  169, 1903,  519,  389, 1029, 2810,
                                2909, 1779, 1913, 3001, 2188,   43,  939, 2110, 1520, 2732,  841, 2439,
                                1706, 2531, 1243, 1174,   77, 2263,  768,  631, 1699, 2376,  189,  667,
                                2256, 2118, 1774, 2469, 1023, 1627,  536, 1387, 2494,  174, 1071, 1133,
                                1935,   82, 2007,  391,  309,  603,  369,  907,  884, 1463, 2368, 1500,
                                1919,  920, 2718,  796, 2213, 1391, 2963,  246, 1125, 2746, 2928,  782,
                                757,   64, 1316,   26, 1767,   27, 1386,  221,  682,  951, 2240,  947,
                                1093, 1424,  718, 1912,  317, 1917, 2891,  181, 2039, 1301, 1214, 2479,
                                471, 2086,  396, 2435, 1216, 2957, 2040, 1166,  622, 2900, 1435,  683,
                                2470, 2168,  543, 1173,  113, 2604, 2233, 2084, 1930,   53, 2975, 1915,
                                529,  577, 2058, 1085, 2204, 2616,  473, 2241,  613, 1231, 1393, 1385,
                                191,  851, 1144, 2782, 2756,  673, 1170, 2386,  617,  811, 1240, 2485,
                                1573, 1632, 1128, 2154,    1, 1352, 2603, 2839,  178,  546,  976, 1282,
                                2849, 2475, 1269,  122, 2682,  446, 2878,   58, 2505,  250,  562, 2791,
                                2864, 1855, 1775, 2920,  629, 2896,  706, 2589, 1203, 2599,  410, 1711,
                                2777, 1096,  360, 2809, 2985,  254, 2158, 2796, 1438, 2571,  699,  871,
                                954,  865, 1107, 2974,  559,  676])
        return batch


    def forward(self,neighbor_aggr_feature_per_metapath):
        if False: 
            batch = self.test_batch()
        
        semantic_fusion_per_nodetype_feature = self.calc_submetapath_semantic_fusion_feature(neighbor_aggr_feature_per_metapath=neighbor_aggr_feature_per_metapath,node_slices=self.node_slices,echo=False)
        
        for node_type in self.ntypes:
            self.hetero_g.nodes[node_type].data[node_type] = semantic_fusion_per_nodetype_feature[node_type].clone().to("cpu")
            gc.collect()
            
        self.hetero_g = self.hg_propagate_feat_dgl(g=self.hetero_g,num_hops=int(self.cfg['num_hop']),max_length=int(self.cfg['num_hop'])+1,echo=False,tgt_type_metapath_aggr=True)
        submetapath_feature_dict = {}
        feat_keys = list(self.hetero_g.nodes[self.cfg['tgt_type']].data.keys())
        
        for k in feat_keys:                
            submetapath_feature_dict[k] = self.hetero_g.nodes[self.cfg['tgt_type']].data.pop(k).to("cuda")
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
        submetapath_emmbedding_feature = {}
        for node_id, metapaths in tqdm(neighbor_aggr_feature_per_metapath.items()):
            mean_feature_list = []
            for metapath,feature in metapaths.items():
                lin_feature = self.embeding_concat_ver2[metapath](feature.to("cuda"))
                mean_feature_list.append(lin_feature)
            #------（Semantic Fusion）----
            # ntype = list(metapaths.keys())[0]
            # semantic_fusion_per_node_list = torch.cat(list(metapaths.values()),dim=0)
            # submetapath_emmbedding_feature[node_id] = semantic_fusion_per_node_list  
            semantic_fusion_feature_list_tensor = torch.stack(mean_feature_list)
            semantic_fusion_feature = torch.sum(semantic_fusion_feature_list_tensor,dim=0)
            semantic_fusion_feature = semantic_fusion_feature / len(mean_feature_list)
            
            submetapath_emmbedding_feature[node_id] = semantic_fusion_feature  
            #------（Semantic Fusion）----
        
        if echo: print(submetapath_emmbedding_feature)  
        
        # node type ごとの特徴 tensor
        feature_dict = {}
        for ntype, (start, end) in node_slices.items():
            feature_list = []
            for node_id in range(start,end):
                feature_list.append(submetapath_emmbedding_feature[node_id])
            feature_dict[ntype] = torch.stack(feature_list)

        # 出力する
        if echo:
            for key, value in feature_dict.items():
                print(f'{key}: {value}')
            
        return feature_dict    

class LinearPerSubMetapath(nn.Module):
    '''
        Linear projection per submetapath for feature projection in SeHGNN_ver2.
    '''
    def __init__(self, cin, cout):
        super(LinearPerSubMetapath, self).__init__()
        self.cin = cin #隠れ層
        self.cout = cout #出力層

        self.W = nn.Parameter(torch.randn(self.cin, self.cout))
        self.bias = nn.Parameter(torch.zeros(self.cout))

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return x @ self.W + self.bias

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