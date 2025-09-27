import torch
import math
import os
import pickle
from agent import Agent
import random
import psutil
import os
import time
import numpy as np
import torch.nn as nn
# from torch import _linalg_utils
import torch.nn.functional as F

"""
Model for HMASAC




class Predictor(nn.Module):

    def __init__(self, in_dim, n_cats):
        super(Predictor, self).__init__()
        self.predict = nn.Linear(in_dim, n_cats)
        torch.nn.init.xavier_uniform_(self.predict.weight)
        self.sigma = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, node_emb):
        n_cat = self.predict(node_emb)
        return n_cat


class Readout(nn.Module):

    def __init__(self, in_dim, method="mean"):
        super(Readout, self).__init__()
        self.method = method
        self.linear = nn.Linear(2*in_dim, in_dim)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, node_fea, raw):
        if self.method == "mean":
            return self.linear(torch.cat([node_fea, raw],dim=-1))  



class SpatialHyperedge(nn.Module):

    def __init__(self,in_dim, lb_th, num_node, l2_lamda = 0.001,recons_lamda = 0.2):
        super(SpatialHyperedge, self).__init__()
        self.l2_lamda = l2_lamda
        self.fea_dim = in_dim
        self.recons_lamda = recons_lamda
        # projection matrix
        self.num_node = num_node
        self.r_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim,in_dim))) 
        # reconstruction linear combination
        self.incidence_m = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(num_node, num_node-1)))  
        self.lb_th = lb_th  

    def forward(self, X):
      
        self.incidence_all = torch.zeros(self.num_node, self.num_node) 
        self_node = torch.eye(self.num_node)
        self.recon_loss = None
        batch = len(X)
        X = X.detach()  
        for node_idx in range(self.num_node):
            master_node_fea = X[:,node_idx,:]  
            master_node_proj_fea = torch.matmul(master_node_fea, self.r_proj).reshape(-1, self.fea_dim) 
            slave_node_idx = [i for i in range(self.num_node) if i != node_idx]  
            node_linear_comb = self.incidence_m[node_idx].unsqueeze(0)  
            node_linear_comb = torch.clamp(node_linear_comb, min=self.lb_th)  
            node_linear_comb_mask = node_linear_comb > self.lb_th
         
            # threshold = torch.topk(node_linear_comb.view(-1), int(self.lb_th)).values[-1]
           
            # node_linear_comb_mask = node_linear_comb >= threshold
            node_linear_comb = node_linear_comb.masked_fill(~node_linear_comb_mask, value=torch.tensor(0)) 
            neigh_recon_fea = torch.matmul(node_linear_comb, X[:,slave_node_idx,:]).reshape(batch,-1)  
            self.incidence_all[node_idx][slave_node_idx] = node_linear_comb  
       
            linear_comb_l1 = torch.max(torch.sum(torch.abs(node_linear_comb),dim=0))

            # linear_comb_l1 = torch.linalg.norm(node_linear_comb, ord=1)  
            linear_comb_l2 = torch.sqrt(torch.sum(node_linear_comb.pow(2)))

            # linear_comb_l2 = torch.linalg.norm(node_linear_comb, ord=2)  
            recon_error = torch.sqrt(torch.sum((master_node_proj_fea - neigh_recon_fea).pow(2),dim=-1)).reshape(batch,-1) 
            # recon_error = torch.linalg.norm(master_node_proj_fea - neigh_recon_fea, ord=2)  
                # recon_error = torch.cdist(master_node_proj_fea,neigh_recon_fea,2)
                # linear_comb_l1 = torch.linalg.norm(self.incidence_m[neigh_recon_idx],1)
                # linear_comb_l2 = torch.linalg.norm(self.incidence_m[neigh_recon_idx],2)
            node_recons_loss = recon_error * self.recons_lamda + linear_comb_l1 + self.l2_lamda * linear_comb_l2  # dim(?-1,1)
                # node_recons_loss = recon_error.squeeze()
            if self.recon_loss is None:

                self.recon_loss = node_recons_loss
            else:

                self.recon_loss = torch.cat([self.recon_loss, node_recons_loss], dim=1)

        self.incidence_all = self.incidence_all + self_node 
        return self.recon_loss, self.incidence_all 


class TemporalHyperedge(nn.Module):

    def __init__(self,in_dim, lb_th,num_node, l2_lamda = 0.001,recons_lamda = 0.2):
        super(TemporalHyperedge, self).__init__()
        self.l2_lamda = l2_lamda
        self.recons_lamda = recons_lamda
        # projection matrix
        self.num_node = num_node
        self.r_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim, in_dim)))  
        # reconstruction linear combination
        self.incidence_m = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(num_node,num_node)))  
        self.lb_th = lb_th

    def forward(self, cur, pre):
        # pre/cur.shape(?-1,36,32)
        # self.incidence_all = torch.zeros(self.num_node, self.num_node)
        self_node = torch.eye(self.num_node)
        self.recon_loss = 0
        batch = len(cur)
        cur = cur.detach()
        pre = pre.detach()
        for node_idx in range(self.num_node):
            master_node_fea = cur[:,node_idx,:] 
            master_node_proj_fea = torch.matmul(master_node_fea, self.r_proj).reshape(batch, -1)  
            slave_node_idx = [i for i in range(self.num_node)]  
            node_linear_comb = self.incidence_m[node_idx].unsqueeze(0)  
            node_linear_comb = torch.clamp(node_linear_comb, min=self.lb_th)
            node_linear_comb_mask = node_linear_comb > self.lb_th
            # threshold = torch.topk(node_linear_comb.view(-1), int(self.lb_th)).values[-1]
          
            # node_linear_comb_mask = node_linear_comb >= threshold
            node_linear_comb = node_linear_comb.masked_fill(~node_linear_comb_mask, value=torch.tensor(0))
            neigh_recon_fea = torch.matmul(node_linear_comb, pre).reshape(batch,-1) 
            # self.incidence_all[node_idx][slave_node_idx] = node_linear_comb
            # 改写
            linear_comb_l1 = torch.max(torch.sum(torch.abs(node_linear_comb), dim=0))
            # linear_comb_l1 = torch.linalg.norm(node_linear_comb, ord=1)

            linear_comb_l2 = torch.sqrt(torch.sum(node_linear_comb.pow(2)))
            # linear_comb_l2 = torch.linalg.norm(node_linear_comb, ord=2)

            recon_error = torch.sqrt(torch.sum((master_node_proj_fea - neigh_recon_fea).pow(2),dim=-1)).reshape(batch,-1) 
            # recon_error = torch.linalg.norm(master_node_proj_fea - neigh_recon_fea, ord=2)
                # recon_error = torch.cdist(master_node_proj_fea,neigh_recon_fea,2)
                # linear_comb_l1 = torch.linalg.norm(self.incidence_m[neigh_recon_idx],1)
                # linear_comb_l2 = torch.linalg.norm(self.incidence_m[neigh_recon_idx],2)
            node_recons_loss = recon_error * self.recons_lamda + linear_comb_l1 + self.l2_lamda * linear_comb_l2
                # node_recons_loss = recon_error.squeeze()
            self.recon_loss += node_recons_loss
        # self.incidence_all = self.incidence_all + self_node
        return self.recon_loss, self.incidence_m 



class SpatialHyperedgeMP(nn.Module):

    def __init__(self):
        super(SpatialHyperedgeMP, self).__init__()

    def forward(self,cur,incidence_m):
        # cur.shape(?-1,36,32),incidence_m.shape(?-1,36,32)
        batch = len(cur)
        edge_fea = torch.matmul(incidence_m, cur)  
        edge_degree = torch.sum(incidence_m, dim=-1).reshape(-1, 1)  
        edge_fea_normed = torch.div(edge_fea, edge_degree)
        return edge_fea_normed 



class TemporalHyperedgeMP(nn.Module):

    def __init__(self):
        super(TemporalHyperedgeMP, self).__init__()

    def forward(self, cur, pre, incidence_m):
        # # cur.shape(?-1,36,32),incidence_m.shape(36,36)
        edge_fea = torch.matmul(incidence_m, pre) + cur  
        self_degree = torch.ones(incidence_m.shape[0], 1)  
        edge_degree = torch.sum(incidence_m, dim=1).reshape(-1, 1) + self_degree
        edge_fea_normed = torch.div(edge_fea, edge_degree)  # dim(?-1,36,32)
        return edge_fea_normed  



class HHNodeMP(nn.Module):

    def __init__(self, num_node, in_dim = 32,  drop_rate = 0.3):
        super(HHNodeMP, self).__init__()
        self.node_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim, in_dim)))  
        self.spatial_edge_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim, in_dim)))
        self.temporal_edge_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim, in_dim)))
        self.num_node = num_node
        self.act_1 = nn.Softmax(dim = 0)
        self.in_dim = in_dim
        self.drop = nn.Dropout(drop_rate)
        self.act = nn.ReLU(inplace=True)
        self.theta = nn.Linear(in_dim, in_dim, bias=True)
        torch.nn.init.xavier_uniform_(self.theta.weight)

    def forward(self, cur, spatial_hyperedge_emb, temporal_hyperedge_emb):
        # cur：dim(?-1,36,32),emb:(?-1,36,32)
        rlt = []
        batch = len(cur)
        for node_idx in range(self.num_node):
            node_fea = cur[:,node_idx,:]  
            node_fea = (torch.matmul(node_fea, self.node_proj)).reshape(batch,-1,1)  

            spatial_hyperedge_fea = spatial_hyperedge_emb[:,node_idx,:]  
            temporal_hyperedge_fea = temporal_hyperedge_emb[:,node_idx,:]  
            spatial_hyperedge_fea = torch.matmul(spatial_hyperedge_fea, self.spatial_edge_proj)  
            temporal_hyperedge_fea = torch.matmul(temporal_hyperedge_fea, self.temporal_edge_proj)  

            # hyperedge = torch.vstack([spatial_hyperedge_fea, temporal_hyperedge_fea])  
            
            hyperedge = torch.cat((spatial_hyperedge_fea[:,None,:],temporal_hyperedge_fea[:,None,:]),dim=1)  
            # TODO：attention在这里。
            atten = self.act_1(torch.matmul(hyperedge, node_fea)/math.sqrt(self.in_dim)).reshape(batch,-1, 1)  
            rlt.append(torch.sum(torch.mul(atten, hyperedge), dim=1).reshape(batch,1, -1)) 
        concatenated = torch.stack(rlt, dim=1).reshape(batch,-1,32)  # dim(?-1,36,32)
        return self.drop(self.act(self.theta(concatenated)))  


class TimeBlock(nn.Module):
    def __init__(self, in_dim=32, num_node=36):
        super(TimeBlock, self).__init__()
        self.lb_th =0.4
        self.num_node = 12
        self.spatial_hyperedge_1 = SpatialHyperedge(in_dim,self.lb_th,self.num_node)
        self.spatial_hyperedge_2 = SpatialHyperedge(in_dim, 1.5*self.lb_th, self.num_node)
        #self.spatial_hyperedge_3 = SpatialHyperedge(in_dim, 2 * self.lb_th, self.num_node)
        self.temporal_hyperedge = TemporalHyperedge(in_dim,self.lb_th,self.num_node)
        self.spatial_hyperedge_MP_1 = SpatialHyperedgeMP()
        self.spatial_hyperedge_MP_2 = SpatialHyperedgeMP()
        #elf.spatial_hyperedge_MP_3 = SpatialHyperedgeMP()
        self.temporal_hyperedge_MP = TemporalHyperedgeMP()
        self.node_mp = HHNodeMP(self.num_node)

    def forward(self, pre, cur,pre_vehicle_num,cur_vehicle_num): 
        """
        :param cur: (?-1)*N * d
        :param pred: (?-1)*N * d
        :return: (?-1)*N * d
        """

        spatial_hyperedge_loss_1, spatial_hyperedge_incidence_1 = self.spatial_hyperedge_1(cur)  
        spatial_hyperedge_loss_2, spatial_hyperedge_incidence_2 = self.spatial_hyperedge_2(cur)
        max_num =cur_vehicle_num.max(dim=1, keepdim=False).values.unsqueeze(1) 
        min_num =cur_vehicle_num.min(dim=1, keepdim=False).values.unsqueeze(1)
        diff = (max_num - min_num).clamp(min=1e-8)
        weight_num = cur_vehicle_num - min_num 
        weight = weight_num/diff
        weight1 = weight*0.5+0.25 
        weight2 = torch.ones_like(weight1)
        weight2-=weight1
        #spatial_hyperedge_loss_3, spatial_hyperedge_incidence_3 = self.spatial_hyperedge_3(cur)
        temporal_hyperedge_loss, temporal_hyperedge_incidence = self.temporal_hyperedge(cur, pre)
        spatial_hyperedge_emb_1 = self.spatial_hyperedge_MP_1(cur, spatial_hyperedge_incidence_1) 
        spatial_hyperedge_emb_2 = self.spatial_hyperedge_MP_2(cur, spatial_hyperedge_incidence_2)
        #spatial_hyperedge_emb_3 = self.spatial_hyperedge_MP_3(cur, spatial_hyperedge_incidence_3)
        spatial_hyperedge_emb =spatial_hyperedge_emb_1*weight1+spatial_hyperedge_emb_2*weight2
        spatial_hyperedge_loss =(spatial_hyperedge_loss_1 * weight1.squeeze(2)+spatial_hyperedge_loss_2 * weight2.squeeze(2)).sum(dim=1, keepdim=True)
        #spatial_hyperedge_loss = (spatial_hyperedge_loss_1 + spatial_hyperedge_loss_2).sum(dim=1, keepdim=True)
        temporal_hyperedge_emb = self.temporal_hyperedge_MP(cur, pre, temporal_hyperedge_incidence)  
        node_emb = self.node_mp(cur, spatial_hyperedge_emb, temporal_hyperedge_emb)  
        return node_emb, temporal_hyperedge_loss+spatial_hyperedge_loss  


class HyperModule(nn.Module):
    """
    multi-timestamp training
    """
    def __init__(self, win_size, h_dim=32, n_cats=4, recons_lambda=0.1):
        super(HyperModule, self).__init__()
        self.win_size = win_size #2
        self.time_cursor = TimeBlock()
        self.predictor = Predictor(h_dim, n_cats=n_cats)
        self.readout = Readout(h_dim)
        self.recons_lambda = recons_lambda

    def forward(self, node_fea,vehicle_num): 
        recon_loss = 0
        # pre_node.shape(?-1,36,32），cur_node.shape(?-1,36,32)
        for i in range(self.win_size-1):  
            if i == 0:
                pre_node = node_fea[:-1,:,:]  
                cur_node = node_fea[1:,:,:]
                pre_vehicle_num = vehicle_num[:-1,:,:]
                cur_vehicle_num = vehicle_num[1:, :, :]
                cur_node_emb, r_loss = self.time_cursor(pre_node, cur_node,pre_vehicle_num,cur_vehicle_num) 
                recon_loss += r_loss
            else:
                cur_node = node_fea[i + 1].contiguous()
                pre_node = cur_node_emb.contiguous()
                cur_node_emb,r_loss = self.time_cursor(pre_node, cur_node)
                recon_loss += r_loss
        graph_emb = self.readout(cur_node_emb, node_fea[1:,:,:])  
        logits = self.predictor(graph_emb) 
        
        return logits, recon_loss*self.recons_lambda  


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim=20, hidden_dim=8, action_dim=4)
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):  
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim= -1)  


class QValueNet(torch.nn.Module):
    
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 20-->32
       
        self.HyperModule = HyperModule(win_size=2)
        # self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x_pair):  
        x = F.relu(self.fc1(x_pair))  # dim(?,36,32)
        vehicle_num = x_pair[..., :12].sum(dim=2, keepdim=True)
        total_logits, total_r_loss = self.HyperModule(x,vehicle_num) # -->dim(?-1,36,4) ，dim(?-1,1)
        print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        return total_logits, total_r_loss


class HMASACNet(nn.Module):
    def __init__(self, state_dim=20, hidden_dim=8, action_dim=4, actor_lr=1e-3, critic_lr=1e-2,
                 alpha_lr=1e-2, target_entropy=-1, tau=0.005, gamma=0.98):
        super(HMASACNet,self).__init__()
        
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim)
      
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim)
       
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim,
                                         action_dim)  
        self.target_critic_2 = QValueNet(state_dim, hidden_dim,
                                         action_dim)  
        
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy  
        self.gamma = gamma
        self.tau = tau
        self.alpha_hyper = 0.001 


class HMaSACAgent(Agent):
    def __init__(self,
                 dic_agent_conf=None,
                 dic_traffic_env_conf=None,
                 dic_path=None,
                 cnt_round=None,
                 best_round=None,
                 intersection_id="0",
                 bar_round=None
                 ):

        super(HMaSACAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)  

      
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_actions = len(
            self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])  
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))  
        self.memory = self.build_memory()  
        self.round_cur = cnt_round
       
        state_dim=20
        hidden_dim=32
        action_dim=4
        actor_lr = 1e-4
        critic_lr = 1e-3
        alpha_lr = 1e-3
        target_entropy = -0.5  
        tau = 0.005
        gamma = 0.98

        if cnt_round == 0:
            # initialization
            self.HmaSACNet = HMASACNet(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma)  

            if os.listdir(self.dic_path["PATH_TO_MODEL"]):  
                self.HmaSACNet.load_state_dict(torch.load(
                    os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_{0}.pth".format(intersection_id))))  
            # self.HmaSACNet_bar = self.build_network_from_copy(self.HmaSACNet)  
        else:
           
            try:
                if best_round:
                    pass
                else:
                    self.load_network("round_{0}_inter_{1}".format(max(cnt_round - 1,0), self.intersection_id)) 
            except:
                print("fail to load network, current round: {0}".format(cnt_round))

        if os.path.exists(os.path.join(self.dic_path["PATH_TO_MODEL"],"round_-1_inter_{0}.pth".format(intersection_id))):
            # the 0-th model is pretrained model
            self.dic_agent_conf["EPSILON"] = self.dic_agent_conf["MIN_EPSILON"]
            print('round%d, EPSILON:%.4f' % (cnt_round, self.dic_agent_conf["EPSILON"]))
        else:
            decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"],
                                                                   cnt_round)  
            self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])  



    def action_att_predict(self, state, total_features=[],  bar=False):
        # state:cur_phase,lane_num_v,adjacency_matrix,adjacency_matrix_lane
        batch_size = len(state)  
        # state = torch.tensor([state], dtype=torch.float)
        if total_features == []:
            total_features = list()
            for i in range(batch_size):
                feature = []  
                for j in range(self.num_agents):
                    observation = []  
                    for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                        if 'adjacency' in feature_name:
                            continue
                        if feature_name == "cur_phase":
                            if len(state[i][j][feature_name]) == 1:
                                # choose_action
                                observation.extend(
                                    self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']]
                                    [state[i][j][feature_name][0]])
                            else:
                                observation.extend(state[i][j][feature_name])
                        elif feature_name == "lane_num_vehicle":
                            observation.extend(state[i][j][feature_name])
                    feature.append(observation) 
                total_features.append(feature)  
            # feature:[agents,feature]
            total_features = np.reshape(np.array(total_features), [batch_size, self.num_agents, -1]) 
            # adj:[agent,neighbors]

        total_features = torch.tensor(total_features, dtype=torch.float)
        all_output = self.HmaSACNet.actor(total_features) 
        action = all_output

        # out: [batch,agent,action]
        if len(action) > 1:
          
            return total_features 

        # [batch,agent,1]
        action = action.detach().numpy()  
        max_action = np.expand_dims(np.argmax(action, axis=-1), axis=-1)  
        random_action = np.reshape(np.random.randint(self.num_actions, size=1 * self.num_agents),
                                   (1, self.num_agents, 1))
        # [batch,agent,2]
        possible_action = np.concatenate([max_action, random_action], axis=-1)  
        selection = np.random.choice(
            [0, 1],
            size=batch_size * self.num_agents,
            p=[1 - self.dic_agent_conf["EPSILON"], self.dic_agent_conf["EPSILON"]])
        act = possible_action.reshape((batch_size * self.num_agents, 2))[
            np.arange(batch_size * self.num_agents), selection]
        act = np.reshape(act, (batch_size, self.num_agents))
        return act # dim(1,36,)


    def choose_action(self, count, state):
       
        act = self.action_att_predict([state]) 
        return act[0]  # dim(36,1)


    def prepare_Xs_Y(self, memory, dic_exp_conf):
        
        ind_end = len(memory)  
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting
        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            sample_slice = memory  
        # forget
        else:
            
            ind_sta = max(0, ind_end - 1000)  
            # ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])  
            memory_after_forget = memory[ind_sta: ind_end]  
            print("memory size after forget:", len(memory_after_forget))

         
            sample_size = min(20, len(memory_after_forget))  
           
            start = random.randint(0,len(memory_after_forget)-sample_size)
            sample_slice = memory_after_forget[start:start + sample_size]
            
            print("memory samples number:", sample_size)

   
        _state = []
        _next_state = []
        _action = []
        _reward = []

      
        for i in range(len(sample_slice)):
            _state.append([])
            _next_state.append([])
            _action.append([])
            _reward.append([])
            for j in range(self.num_agents):
                state, action, next_state, reward, _ = sample_slice[i][j]
                _state[i].append(state)
                _next_state[i].append(next_state)
                _action[i].append(action)
                _reward[i].append(reward)


     
      
        _features = self.action_att_predict(_state)
        _next_features = self.action_att_predict(_next_state)  

       
        self.transition_dict = {
            'states':_features,
            'actions':_action,
            'next_states':_next_features,
            'rewards':_reward,
            'dones':None
        }
        return


    def build_memory(self):

        return []

   
    def load_network(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        self.HmaSACNet = torch.load(
            os.path.join(file_path, "%s.pth" % file_name))
        print("succeed in loading model %s"%file_name)


    def save_network(self, file_name):
        torch.save(self.HmaSACNet, os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.pth" % file_name))


    def train_network(self, dic_exp_conf):

        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            epochs = 1000
        else:
            epochs = self.dic_agent_conf["EPOCHS"]  # epchs=100
        # batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))  # batch_size为20
        #

        for i in range(50):
            self.update(self.transition_dict,i)

       

    def calc_target(self, rewards, next_states, dones=0):
        next_probs = self.HmaSACNet.actor(next_states)  
        next_probs = next_probs[1:]
        rewards = rewards[1:]
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=-1, keepdim=True) 
        q1_value, _ = self.HmaSACNet.target_critic_1(next_states) 
        q2_value, _ = self.HmaSACNet.target_critic_2(next_states)
    
        # min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=-1, keepdim=True)  # dim(?-1,36,1)
      
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=-1,keepdim=True)

        next_value = min_qvalue + self.HmaSACNet.log_alpha.exp() * entropy
        td_target = rewards + self.HmaSACNet.gamma * next_value * (1 - dones)
        return td_target  # dim(?-1,36,1)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.HmaSACNet.tau) + param.data * self.HmaSACNet.tau)


    def update(self, transition_dict, i):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)  # dim(1000,36,20)
        actions = torch.tensor(transition_dict['actions']).view(-1, self.num_agents ,1) 
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, self.num_agents, 1)  
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)  # dim(1000,36,20)
        # dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)
 
        td_target = self.calc_target(rewards, next_states)  # dim(?-1,36,1)
        actions = actions[1:]
      
        critic_1_q_values, critic_1_recon_loss = self.HmaSACNet.critic_1(states)  
        critic_q_1_a = critic_1_q_values.gather(-1, actions) 
        critic_1_loss = ((1-self.HmaSACNet.alpha_hyper)*torch.mean(F.mse_loss(critic_q_1_a, td_target.detach())) +
                         self.HmaSACNet.alpha_hyper * torch.mean(critic_1_recon_loss))
        critic_2_q_values, critic_2_recon_loss = self.HmaSACNet.critic_2(states)
        critic_q_2_a = critic_2_q_values.gather(-1, actions)
        critic_2_loss = ((1-self.HmaSACNet.alpha_hyper) * torch.mean(F.mse_loss(critic_q_2_a, td_target.detach())) +
                         self.HmaSACNet.alpha_hyper * torch.mean(critic_2_recon_loss)) 
        self.HmaSACNet.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.HmaSACNet.critic_1_optimizer.step()
        self.HmaSACNet.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.HmaSACNet.critic_2_optimizer.step()
        print('round:'+ str(self.round_cur) + ' Epoch:' + str(i+1) + ' critic_1_loss:' + str(critic_1_loss))
        print('round:'+ str(self.round_cur) + ' Epoch:' + str(i+1) + ' critic_2_loss:' + str(critic_2_loss))

      
        probs = self.HmaSACNet.actor(states)
        probs = probs[1:]  
        log_probs = torch.log(probs + 1e-8)
     
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)  
        q1_value, _ = self.HmaSACNet.critic_1(states)
        q2_value, _ = self.HmaSACNet.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=-1, keepdim=True)  
        actor_loss = torch.mean(-self.HmaSACNet.log_alpha.exp() * entropy - min_qvalue)
        self.HmaSACNet.actor_optimizer.zero_grad()
        actor_loss.backward()
        print('round:'+ str(self.round_cur) +' Epoch:' + str(i+1) +  ' actor_loss:' + str(actor_loss))

        self.HmaSACNet.actor_optimizer.step()


        alpha_loss = torch.mean((entropy - self.HmaSACNet.target_entropy).detach() * self.HmaSACNet.log_alpha.exp())
        self.HmaSACNet.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.HmaSACNet.log_alpha_optimizer.step()

        self.soft_update(self.HmaSACNet.critic_1, self.HmaSACNet.target_critic_1)
        self.soft_update(self.HmaSACNet.critic_2, self.HmaSACNet.target_critic_2)



