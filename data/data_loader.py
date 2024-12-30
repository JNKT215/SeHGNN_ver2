# Modified from https://github.com/THUDM/HGB/blob/master/NC/benchmark/scripts/data_loader.py, aiming to load hgb's datasets.

import os,torch,dgl
from torch_sparse import SparseTensor
import numpy as np
import scipy.sparse as sp
from collections import Counter, defaultdict
from sklearn.metrics import f1_score


class data_loader:
    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            pass
        elif os.path.exists(path+'.zip'):
            os.system("unzip {} -d {}".format(path+'.zip', os.path.join(*path.split('/')[:-1])))
        else:
            assert False, ('Warning: HGB node classification datasets not downloaded'
                ', please download them from HGB repository `https://github.com/THUDM/HGB`'
                ' and put them under `./data/`')

        self.nodes = self.load_nodes()
        self.links = self.load_links()
        self.labels_train = self.load_labels('label.dat')
        self.labels_test = self.load_labels('label.dat.test')

    def get_sub_graph(self, node_types_tokeep):
        """
        node_types_tokeep is a list or set of node types that you want to keep in the sub-graph
        We only support whole type sub-graph for now.
        This is an in-place update function!
        return: old node type id to new node type id dict, old edge type id to new edge type id dict
        """
        keep = set(node_types_tokeep)
        new_node_type = 0
        new_node_id = 0
        new_nodes = {'total':0, 'count':Counter(), 'attr':{}, 'shift':{}}
        new_links = {'total':0, 'count':Counter(), 'meta':{}, 'data':defaultdict(list)}
        new_labels_train = {'num_classes':0, 'total':0, 'count':Counter(), 'data':None, 'mask':None}
        new_labels_test = {'num_classes':0, 'total':0, 'count':Counter(), 'data':None, 'mask':None}
        old_nt2new_nt = {}
        old_idx = []
        for node_type in self.nodes['count']:
            if node_type in keep:
                nt = node_type
                nnt = new_node_type
                old_nt2new_nt[nt] = nnt
                cnt = self.nodes['count'][nt]
                new_nodes['total'] += cnt
                new_nodes['count'][nnt] = cnt
                new_nodes['attr'][nnt] = self.nodes['attr'][nt]
                new_nodes['shift'][nnt] = new_node_id
                beg = self.nodes['shift'][nt]
                old_idx.extend(range(beg, beg+cnt))

                cnt_label_train = self.labels_train['count'][nt]
                new_labels_train['count'][nnt] = cnt_label_train
                new_labels_train['total'] += cnt_label_train
                cnt_label_test = self.labels_test['count'][nt]
                new_labels_test['count'][nnt] = cnt_label_test
                new_labels_test['total'] += cnt_label_test

                new_node_type += 1
                new_node_id += cnt

        new_labels_train['num_classes'] = self.labels_train['num_classes']
        new_labels_test['num_classes'] = self.labels_test['num_classes']
        for k in ['data', 'mask']:
            new_labels_train[k] = self.labels_train[k][old_idx]
            new_labels_test[k] = self.labels_test[k][old_idx]

        old_et2new_et = {}
        new_edge_type = 0
        for edge_type in self.links['count']:
            h, t = self.links['meta'][edge_type]
            if h in keep and t in keep:
                et = edge_type
                net = new_edge_type
                old_et2new_et[et] = net
                new_links['total'] += self.links['count'][et]
                new_links['count'][net] = self.links['count'][et]
                new_links['meta'][net] = tuple(map(lambda x:old_nt2new_nt[x], self.links['meta'][et]))
                new_links['data'][net] = self.links['data'][et][old_idx][:, old_idx]
                new_edge_type += 1

        self.nodes = new_nodes
        self.links = new_links
        self.labels_train = new_labels_train
        self.labels_test = new_labels_test
        return old_nt2new_nt, old_et2new_et

    def get_meta_path(self, meta=[]):
        """
        Get meta path matrix
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a sparse matrix with shape [node_num, node_num]
        """
        ini = sp.eye(self.nodes['total'])
        meta = [self.get_edge_type(x) for x in meta]
        for x in meta:
            ini = ini.dot(self.links['data'][x]) if x >= 0 else ini.dot(self.links['data'][-x - 1].T)
        return ini

    def dfs(self, now, meta, meta_dict):
        if len(meta) == 0:
            meta_dict[now[0]].append(now)
            return
        th_mat = self.links['data'][meta[0]] if meta[0] >= 0 else self.links['data'][-meta[0] - 1].T
        th_node = now[-1]
        for col in th_mat[th_node].nonzero()[1]:
            self.dfs(now+[col], meta[1:], meta_dict)

    def get_full_meta_path(self, meta=[], symmetric=False):
        """
        Get full meta path for each node
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a dict of list[list] (key is node_id)
        """
        meta = [self.get_edge_type(x) for x in meta]
        if len(meta) == 1:
            meta_dict = {}
            start_node_type = self.links['meta'][meta[0]][0] if meta[0]>=0 else self.links['meta'][-meta[0]-1][1]
            for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                meta_dict[i] = []
                self.dfs([i], meta, meta_dict)
        else:
            meta_dict1 = {}
            meta_dict2 = {}
            mid = len(meta) // 2
            meta1 = meta[:mid]
            meta2 = meta[mid:]
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0]>=0 else self.links['meta'][-meta1[0]-1][1]
            for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                meta_dict1[i] = []
                self.dfs([i], meta1, meta_dict1)
            start_node_type = self.links['meta'][meta2[0]][0] if meta2[0]>=0 else self.links['meta'][-meta2[0]-1][1]
            for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                meta_dict2[i] = []
            if symmetric:
                for k in meta_dict1:
                    paths = meta_dict1[k]
                    for x in paths:
                        meta_dict2[x[-1]].append(list(reversed(x)))
            else:
                for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                    self.dfs([i], meta2, meta_dict2)
            meta_dict = {}
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0]>=0 else self.links['meta'][-meta1[0]-1][1]
            for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                meta_dict[i] = []
                for beg in meta_dict1[i]:
                    for end in meta_dict2[beg[-1]]:
                        meta_dict[i].append(beg + end[1:])
        return meta_dict

    def gen_file_for_macnemar(self, test_idx, label,true_label, file_name, mode='bi'):
        if test_idx.shape[0] != label.shape[0]:
            return
        if mode == 'multi':
            multi_label=[]
            for i in range(label.shape[0]):
                label_list = [str(j) for j in range(label[i].shape[0]) if label[i][j]==1]
                multi_label.append(' '.join(label_list))
            label=multi_label

            multi_true_label,true_label =[], np.array(true_label.to("cpu"))
            for i in range(true_label.shape[0]):
                label_list = [str(j) for j in range(true_label[i].shape[0]) if true_label[i][j]==1]
                multi_true_label.append(' '.join(label_list))
            true_label = multi_true_label
        elif mode=='bi':
            label = np.array(label)
            true_label = np.array(true_label.to("cpu"))
        else:
            return
        with open(file_name, "w") as f:
            for nid, l,true_l in zip(test_idx, label,true_label):
                f.write(f"{nid},{l},{true_l}\n")

    def gen_file_for_evaluate(self, test_idx, label, file_name, mode='bi'):
        if test_idx.shape[0] != label.shape[0]:
            return
        if mode == 'multi':
            multi_label=[]
            for i in range(label.shape[0]):
                label_list = [str(j) for j in range(label[i].shape[0]) if label[i][j]==1]
                multi_label.append(','.join(label_list))
            label=multi_label
        elif mode=='bi':
            label = np.array(label)
        else:
            return
        with open(file_name, "w") as f:
            for nid, l in zip(test_idx, label):
                f.write(f"{nid}\t\t{self.get_node_type(nid)}\t{l}\n")

    def evaluate(self, pred):
        y_true = self.labels_test['data'][self.labels_test['mask']]
        micro = f1_score(y_true, pred, average='micro')
        macro = f1_score(y_true, pred, average='macro')
        result = {
            'micro-f1': micro,
            'macro-f1': macro
        }
        return result

    def load_labels(self, name):
        """
        return labels dict
            num_classes: total number of labels
            total: total number of labeled data
            count: number of labeled data for each node type
            data: a numpy matrix with shape (self.nodes['total'], self.labels['num_classes'])
            mask: to indicate if that node is labeled, if False, that line of data is masked
        """
        labels = {'num_classes':0, 'total':0, 'count':Counter(), 'data':None, 'mask':None}
        nc = 0
        mask = np.zeros(self.nodes['total'], dtype=bool)
        data = [None for i in range(self.nodes['total'])]
        with open(os.path.join(self.path, name), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                node_id, node_name, node_type, node_label = int(th[0]), th[1], int(th[2]), list(map(int, th[3].split(',')))
                for label in node_label:
                    nc = max(nc, label+1)
                mask[node_id] = True
                data[node_id] = node_label
                labels['count'][node_type] += 1
                labels['total'] += 1
        labels['num_classes'] = nc
        new_data = np.zeros((self.nodes['total'], labels['num_classes']), dtype=int)
        for i,x in enumerate(data):
            if x is not None:
                for j in x:
                    new_data[i, j] = 1
        labels['data'] = new_data
        labels['mask'] = mask
        return labels

    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i]+self.nodes['count'][i]:
                return i

    def get_edge_type(self, info):
        if type(info) is int or len(info) == 1:
            return info
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return i
        info = (info[1], info[0])
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return -i - 1
        raise Exception('No available edge type')

    def get_edge_info(self, edge_id):
        return self.links['meta'][edge_id]

    def list_to_sp_mat(self, li):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i,j)), shape=(self.nodes['total'], self.nodes['total'])).tocsr()

    def load_links(self):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total'], nodes['total'])
        """
        links = {'total':0, 'count':Counter(), 'meta':{}, 'data':defaultdict(list)}
        with open(os.path.join(self.path, 'link.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(th[2]), float(th[3])
                if r_id not in links['meta']:
                    h_type = self.get_node_type(h_id)
                    t_type = self.get_node_type(t_id)
                    links['meta'][r_id] = (h_type, t_type)
                links['data'][r_id].append((h_id, t_id, link_weight))
                links['count'][r_id] += 1
                links['total'] += 1
        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = self.list_to_sp_mat(links['data'][r_id])
        links['data'] = new_data
        return links

    def load_nodes(self):
        """
        return nodes dict
            total: total number of nodes
            count: a dict of int, number of nodes for each type
            attr: a dict of np.array (or None), attribute matrices for each type of nodes
            shift: node_id shift for each type. You can get the id range of a type by
                        [ shift[node_type], shift[node_type]+count[node_type] )
        """
        nodes = {'total':0, 'count':Counter(), 'attr':{}, 'shift':{}}
        with open(os.path.join(self.path, 'node.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                if len(th) == 4:
                    # Then this line of node has attribute
                    node_id, node_name, node_type, node_attr = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    node_attr = list(map(float, node_attr.split(',')))
                    nodes['count'][node_type] += 1
                    nodes['attr'][node_id] = node_attr
                    nodes['total'] += 1
                elif len(th) == 3:
                    # Then this line of node doesn't have attribute
                    node_id, node_name, node_type = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    nodes['count'][node_type] += 1
                    nodes['total'] += 1
                else:
                    raise Exception("Too few information to parse!")
        shift = 0
        attr = {}
        for i in range(len(nodes['count'])):
            nodes['shift'][i] = shift
            if shift in nodes['attr']:
                mat = []
                for j in range(shift, shift+nodes['count'][i]):
                    mat.append(nodes['attr'][j])
                attr[i] = np.array(mat)
            else:
                attr[i] = None
            shift += nodes['count'][i]
        nodes['attr'] = attr
        return nodes


class HeteroDataSet():
    def __init__(self,cfg,root):
        # data_load
        self.cfg = cfg
        self.root = root
        #pre-processing neighbor aggr
        self.raw_feats = {}
        self.feats = {}
        self.label_feats = {}
        self.neighbor_aggr_feature_per_metapath ={}
        #metapath_keys(feature・label_keys)
        self.feat_keys = []
        self.label_feat_keys = []
        #feature_dim_per_metapath
        self.data_size = None
        #device type
        self.prop_device = 'cpu'
        self.store_device = 'cpu'
        
                
        if self.cfg['dataset'] in ['DBLP', 'ACM', 'IMDB','Freebase']: #HGB dataset
            self.g, self.adjs, self.init_labels, self.num_classes, self.dl, self.trainval_nid, self.test_nid = self.load_hgb_dataset()
            # row_normarize_adj_matrix
            for k in self.adjs.keys():
                self.adjs[k].storage._value = None
                self.adjs[k].storage._value = torch.ones(self.adjs[k].nnz()) / self.adjs[k].sum(dim=-1)[self.adjs[k].storage.row()]
            self.labels = self.init_labels.clone()
            
            print(f"dataset: {self.cfg['dataset']}")
            for etype, adj in self.adjs.items():
                print(f"etype:{etype}, density:{adj.density() * 100:.2f}%")

        if self.cfg['dataset'] == "DBLP":
            self.tgt_type = 'A'
            self.node_dict = {"A": 0, "P": 1, "T": 2,"V":3}
            self.edge_type = {0: "PA", 1: "AP", 2: "PV", 3: "VP", 4: "PT", 5: "TP"}
            self.next_type = self.get_next_edge_type()
            self.extra_metapath = []
        elif self.cfg['dataset'] == "IMDB":
            self.tgt_type = 'M'
            self.node_dict = {"M": 0, "D": 1, "A": 2, "K": 3}
            self.edge_type = {0: "MD", 1: "DM", 2: "MA", 3: "AM", 4: "MK", 5: "KM"}
            self.next_type = self.get_next_edge_type()
            self.extra_metapath = []
        elif self.cfg['dataset'] == "ACM":
            self.tgt_type = 'P'
            if not cfg['ACM_keep_F']:
                self.node_dict = {"P": 0, "A": 1, "C": 2}
                self.edge_type = {0: "PP", 1: "PA", 2: "AP", 3: "PC", 4: "CP"}     
            else:
                self.node_dict = {"P": 0, "A": 1, "C": 2, "T": 3}
                self.edge_type = {0: "PP", 1: "PA", 2: "AP", 3: "PC", 4: "CP", 5: "TP", 6: "PT"}               
            self.next_type = self.get_next_edge_type()
            self.extra_metapath = []
        elif self.cfg['dataset'] =='Freebase':
            self.tgt_type = '0'
            self.prop_device   = f'cuda:{self.cfg.gpu}' if not self.cfg.cpu else 'cpu'
            self.raw_meta_adjs = {}
            self.node_dicts = {str(i): i for i in range(8)}
            # self.node_dict = {'B': 0, 'F': 1, 'M': 2, 'S': 3, 'P': 4, 'L': 5, 'O': 6}
            # self.edge_type = {0: 'BB', 1: 'BF', 2: 'BS', 3: 'BL', 4: 'BO', 5: 'FF', 6: 'MB', 7: 'MF', 8: 'MM', 9: 'MS', 10: 'ML', 11: 'SF', 12: 'SS', 13: 'SL', 14: 'PB', 15: 'PF', 16: 'PM', 17: 'PS', 18: 'PP', 19: 'PL', 20: 'PO', 21: 'LF', 22: 'LL', 23: 'OF', 24: 'OM', 25: 'OS', 26: 'OL', 27: 'OO', 28: 'OB', 29: 'BM'}
            self.next_type = self.get_next_edge_type()
            self.extra_metapath = []         
            
            if not os.path.exists('./Freebase_adjs'):
                os.makedirs('./Freebase_adjs')
            self.num_tgt_nodes = self.dl.nodes['count'][0]            
        else:
            assert 0
            
        self.extra_metapath = [ele for ele in self.extra_metapath if len(ele) > self.cfg['num_hop'] + 1]        
        
        if self.cfg['model'] == "SeHGNNver2":
            self.swap_node_dict = {v: k for k, v in self.node_dict.items()}
            self.node_slices = {self.swap_node_dict[n_type_index]:(self.dl.nodes['shift'][n_type_index], self.dl.nodes['shift'][n_type_index] + self.dl.nodes['count'][n_type_index]  ) for n_type_index in range(len(self.node_dict))}
            self.ntype_features = {ntype:self.g.ndata[ntype][ntype].clone() for ntype in self.node_dict}
            if self.cfg['dataset'] == "ACM":
                self.total_nodes = self.dl.nodes['total'] - self.dl.nodes['count'][3]
            else:
                self.total_nodes = self.dl.nodes['total']
            self.neighbor_aggr_feature_per_metapath = None
            
    def get_next_edge_type(self):
        next_type = {}
        # エッジタイプごとに処理
        for edge_id, edge_type in self.edge_type.items():
            next_types = []
            # エッジタイプの最後のノードを取得
            last_node = edge_type[-1]
            # 最後のノードに接続される可能性のあるノードを見つける
            for node in self.node_dict.keys():
                if node == last_node:
                    # 次のエッジタイプを見つける
                    for next_edge_id, next_edge_type in self.edge_type.items():
                        if next_edge_type[0] == node:
                            next_types.append(next_edge_id)
            next_type[edge_id] = next_types
        print(next_type)
        
        return next_type
        
    def get_training_setup(self):
        val_ratio = 0.2
        train_nid = self.trainval_nid.copy()
        np.random.shuffle(train_nid)
        split = int(train_nid.shape[0]*val_ratio)
        val_nid = train_nid[:split]
        train_nid = train_nid[split:]
        train_nid = np.sort(train_nid)
        val_nid = np.sort(val_nid)

        train_node_nums = len(train_nid)
        valid_node_nums = len(val_nid)
        test_node_nums = len(self.test_nid)
        trainval_point = train_node_nums
        valtest_point = trainval_point + valid_node_nums
        print(f'#Train {train_node_nums}, #Val {valid_node_nums}, #Test {test_node_nums}')

        labeled_nid = np.concatenate((train_nid, val_nid, self.test_nid))
        labeled_num_nodes = len(labeled_nid)
        num_nodes = self.dl.nodes['count'][0]

        if labeled_num_nodes < num_nodes:
            flag = np.ones(num_nodes, dtype=bool)
            flag[train_nid] = 0
            flag[val_nid] = 0
            flag[self.test_nid] = 0
            extra_nid = np.where(flag)[0]
            print(f'Find {len(extra_nid)} extra nid for dataset {self.cfg.dataset}')
        else:
            extra_nid = np.array([])
        
        self.num_nodes = num_nodes
        self.train_nid = train_nid
        self.val_nid = val_nid
        self.trainval_point = trainval_point
        self.labeled_num_nodes = labeled_num_nodes
        self.labeled_nid = labeled_nid
        self.extra_nid = extra_nid
        self.valtest_point = valtest_point
    
    def load_hgb_dataset(self):
        dl = data_loader(path=self.root)

        # use one-hot index vectors for nods with no attributes
        # === feats ===
        features_list = []
        for i in range(len(dl.nodes['count'])):
            th = dl.nodes['attr'][i]
            if th is None:
                features_list.append(torch.eye(dl.nodes['count'][i]))
            else:
                features_list.append(torch.FloatTensor(th))

        idx_shift = np.zeros(len(dl.nodes['count'])+1, dtype=np.int32)
        for i in range(len(dl.nodes['count'])):
            idx_shift[i+1] = idx_shift[i] + dl.nodes['count'][i]

        # === labels ===
        num_classes = dl.labels_train['num_classes']
        init_labels = np.zeros((dl.nodes['count'][0], num_classes), dtype=int)

        trainval_nid = np.nonzero(dl.labels_train['mask'])[0]
        test_nid = np.nonzero(dl.labels_test['mask'])[0]

        init_labels[trainval_nid] = dl.labels_train['data'][trainval_nid]
        init_labels[test_nid] = dl.labels_test['data'][test_nid]
        if self.cfg.dataset != 'IMDB':
            init_labels = init_labels.argmax(axis=1)
        init_labels = torch.LongTensor(init_labels)

        # === adjs ===
        # print(dl.nodes['attr'])
        # for k, v in dl.nodes['attr'].items():
        #     if v is None: print('none')
        #     else: print(v.shape)
        adjs = [] if self.cfg.dataset != 'Freebase' else {}
        for i, (k, v) in enumerate(dl.links['data'].items()):
            v = v.tocoo()
            src_type_idx = np.where(idx_shift > v.col[0])[0][0] - 1
            dst_type_idx = np.where(idx_shift > v.row[0])[0][0] - 1
            row = v.row - idx_shift[dst_type_idx]
            col = v.col - idx_shift[src_type_idx]
            sparse_sizes = (dl.nodes['count'][dst_type_idx], dl.nodes['count'][src_type_idx])
            adj = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=sparse_sizes)
            if self.cfg.dataset == 'Freebase':
                name = f'{dst_type_idx}{src_type_idx}'
                assert name not in adjs
                adjs[name] = adj
            else:
                adjs.append(adj)
                print(adj)

        if self.cfg.dataset == 'DBLP':
            # A* --- P --- T
            #        |
            #        V
            # author: [4057, 334]
            # paper : [14328, 4231]
            # term  : [7723, 50]
            # venue(conference) : None
            A, P, T, V = features_list
            AP, PA, PT, PV, TP, VP = adjs

            new_edges = {}
            ntypes = set()
            etypes = [ # src->tgt
                ('P', 'P-A', 'A'),
                ('A', 'A-P', 'P'),
                ('T', 'T-P', 'P'),
                ('V', 'V-P', 'P'),
                ('P', 'P-T', 'T'),
                ('P', 'P-V', 'V'),
            ]
            for etype, adj in zip(etypes, adjs):
                stype, rtype, dtype = etype
                dst, src, _ = adj.coo()
                src = src.numpy()
                dst = dst.numpy()
                new_edges[(stype, rtype, dtype)] = (src, dst)
                ntypes.add(stype)
                ntypes.add(dtype)
            g = dgl.heterograph(new_edges)

            # for i, etype in enumerate(g.etypes):
            #     src, dst, eid = g._graph.edges(i)
            #     adj = SparseTensor(row=dst.long(), col=src.long())
            #     print(etype, adj)

            # g.ndata['feat']['A'] = A # not work
            g.nodes['A'].data['A'] = A
            g.nodes['P'].data['P'] = P
            g.nodes['T'].data['T'] = T
            g.nodes['V'].data['V'] = V
        elif self.cfg.dataset == 'IMDB':
            # A --- M* --- D
            #       |
            #       K
            # movie    : [4932, 3489]
            # director : [2393, 3341]
            # actor    : [6124, 3341]
            # keywords : None
            M, D, A, K = features_list
            MD, DM, MA, AM, MK, KM = adjs
            assert torch.all(DM.storage.col() == MD.t().storage.col())
            assert torch.all(AM.storage.col() == MA.t().storage.col())
            assert torch.all(KM.storage.col() == MK.t().storage.col())

            assert torch.all(MD.storage.rowcount() == 1) # each movie has single director

            new_edges = {}
            ntypes = set()
            etypes = [ # src->tgt
                ('D', 'D-M', 'M'),
                ('M', 'M-D', 'D'),
                ('A', 'A-M', 'M'),
                ('M', 'M-A', 'A'),
                ('K', 'K-M', 'M'),
                ('M', 'M-K', 'K'),
            ]
            for etype, adj in zip(etypes, adjs):
                stype, rtype, dtype = etype
                dst, src, _ = adj.coo()
                src = src.numpy()
                dst = dst.numpy()
                new_edges[(stype, rtype, dtype)] = (src, dst)
                ntypes.add(stype)
                ntypes.add(dtype)
            g = dgl.heterograph(new_edges)

            g.nodes['M'].data['M'] = M
            g.nodes['D'].data['D'] = D
            g.nodes['A'].data['A'] = A
            if self.cfg.num_hop > 2 or self.cfg.two_layer:
                g.nodes['K'].data['K'] = K
        elif self.cfg.dataset == 'ACM':
            # A --- P* --- C
            #       |
            #       K
            # paper     : [3025, 1902]
            # author    : [5959, 1902]
            # conference: [56, 1902]
            # field     : None
            P, A, C, K = features_list
            PP, PP_r, PA, AP, PC, CP, PK, KP = adjs
            row, col = torch.where(P)
            assert torch.all(row == PK.storage.row()) and torch.all(col == PK.storage.col())
            assert torch.all(AP.matmul(PK).to_dense() == A)
            assert torch.all(CP.matmul(PK).to_dense() == C)

            assert torch.all(PA.storage.col() == AP.t().storage.col())
            assert torch.all(PC.storage.col() == CP.t().storage.col())
            assert torch.all(PK.storage.col() == KP.t().storage.col())

            row0, col0, _ = PP.coo()
            row1, col1, _ = PP_r.coo()
            PP = SparseTensor(row=torch.cat((row0, row1)), col=torch.cat((col0, col1)), sparse_sizes=PP.sparse_sizes())
            PP = PP.coalesce()
            PP = PP.set_diag()
            adjs = [PP] + adjs[2:]

            new_edges = {}
            ntypes = set()
            etypes = [ # src->tgt
                ('P', 'P-P', 'P'),
                ('A', 'A-P', 'P'),
                ('P', 'P-A', 'A'),
                ('C', 'C-P', 'P'),
                ('P', 'P-C', 'C'),
            ]
            if self.cfg.ACM_keep_F:
                etypes += [
                    ('K', 'K-P', 'P'),
                    ('P', 'P-K', 'K'),
                ]
            for etype, adj in zip(etypes, adjs):
                stype, rtype, dtype = etype
                dst, src, _ = adj.coo()
                src = src.numpy()
                dst = dst.numpy()
                new_edges[(stype, rtype, dtype)] = (src, dst)
                ntypes.add(stype)
                ntypes.add(dtype)

            g = dgl.heterograph(new_edges)

            g.nodes['P'].data['P'] = P # [3025, 1902]
            g.nodes['A'].data['A'] = A # [5959, 1902]
            g.nodes['C'].data['C'] = C # [56, 1902]
            if self.cfg.ACM_keep_F:
                g.nodes['K'].data['K'] = K # [1902, 1902]
        elif self.cfg.dataset == 'Freebase':
            # 0*: 40402  2/4/7 <-- 0 <-- 0/1/3/5/6
            #  1: 19427  all <-- 1
            #  2: 82351  4/6/7 <-- 2 <-- 0/1/2/3/5
            #  3: 1025   0/2/4/6/7 <-- 3 <-- 1/3/5
            #  4: 17641  4 <-- all
            #  5: 9368   0/2/3/4/6/7 <-- 5 <-- 1/5
            #  6: 2731   0/4 <-- 6 <-- 1/2/3/5/6/7
            #  7: 7153   4/6 <-- 7 <-- 0/1/2/3/5/7
            for i in range(8):
                kk = str(i)
                print(f'==={kk}===')
                for k, v in adjs.items():
                    t, s = k
                    assert s == t or f'{s}{t}' not in adjs
                    if s == kk or t == kk:
                        if s == t:
                            print(k, v.sizes(), v.nnz(),
                                f'symmetric {v.is_symmetric()}; selfloop-ratio: {v.get_diag().sum()}/{v.size(0)}')
                        else:
                            print(k, v.sizes(), v.nnz())

            adjs['00'] = adjs['00'].to_symmetric()
            g = None
        else:
            assert 0

        if self.cfg.dataset == 'DBLP':
            adjs = {'AP': AP, 'PA': PA, 'PT': PT, 'PV': PV, 'TP': TP, 'VP': VP}
        elif self.cfg.dataset == 'ACM':
            adjs = {'PP': PP, 'PA': PA, 'AP': AP, 'PC': PC, 'CP': CP}
        elif self.cfg.dataset == 'IMDB':
            adjs = {'MD': MD, 'DM': DM, 'MA': MA, 'AM': AM, 'MK': MK, 'KM': KM}
        elif self.cfg.dataset == 'Freebase':
            new_adjs = {}
            for rtype, adj in adjs.items():
                dtype, stype = rtype
                if dtype != stype:
                    new_name = f'{stype}{dtype}'
                    assert new_name not in adjs
                    new_adjs[new_name] = adj.t()
            adjs.update(new_adjs)
        else:
            assert 0

        return g, adjs, init_labels, num_classes, dl, trainval_nid, test_nid
