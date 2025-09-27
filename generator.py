import os
import copy
from config import DIC_AGENTS, DIC_ENVS
import time
import sys
from multiprocessing import Process, Pool

# 加载模型，启动模拟器环境，指导模拟并记录结果，执行动作并且得到下一状态，都在这个类里面。
class Generator:
    # 初始化，首先根据模拟器类别（sumo/Cityflow）选择环境，具体的环境配置文件在anon_env.py文件，然后看是否开启预训练，最后初始化每个agent的配置
    def __init__(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, best_round=None):

        self.cnt_round = cnt_round  # 训练次数
        self.cnt_gen = cnt_gen  # 交通控制器的数量
        self.dic_exp_conf = dic_exp_conf
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.agents = [None]*dic_traffic_env_conf['NUM_AGENTS']  # 创造一个长度为智能体数量的列表，且内部元素初始化为None。

        # 是否展开预训练，并且判断是否存在该目录
        if self.dic_exp_conf["PRETRAIN"]:
            self.path_to_log = os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"], "train_round",
                                            "round_" + str(self.cnt_round), "generator_" + str(self.cnt_gen))
        else:
            self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round", "round_"+str(self.cnt_round), "generator_"+str(self.cnt_gen))
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)  

        # 模拟器环境的类型，anon/sumo，形成一个字典
        self.env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                              path_to_log=self.path_to_log,
                              path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
                              dic_traffic_env_conf=self.dic_traffic_env_conf)
        self.env.reset()  # 环境初始化

        # every generator's output
        # generator for pretraining
        # Todo pretrain with intersection_id
        # 开启预训练
        if self.dic_exp_conf["PRETRAIN"]:

            self.agent_name = self.dic_exp_conf["PRETRAIN_MODEL_NAME"]  # 得到预训练的模型名
            self.agent = DIC_AGENTS[self.agent_name](  # 根据智能体名称的简写，得到xxxagent
                dic_agent_conf=self.dic_agent_conf,
                dic_traffic_env_conf=self.dic_traffic_env_conf,
                dic_path=self.dic_path,
                cnt_round=self.cnt_round,
                best_round=best_round,
            )
        # 没有开启预训练
        else:
            start_time = time.time()
            for i in range(dic_traffic_env_conf['NUM_AGENTS']):  # 首先获取智能体数量
                agent_name = self.dic_exp_conf["MODEL_NAME"]  # 得到智能体的名称
                # 针对不同的模型名，对智能体进行不同的配置，colight模型需要事先知道车道邻接信息（指车道间哪些可以直接连接，转向通道设置等结构信息），
                #the CoLight_Signal needs to know the lane adj in advance, from environment's intersection list
                if agent_name=='CoLight_Signal':
                    agent = DIC_AGENTS[agent_name](
                        dic_agent_conf=self.dic_agent_conf,
                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                        dic_path=self.dic_path,
                        cnt_round=self.cnt_round, 
                        best_round=best_round,
                        inter_info=self.env.list_intersection,  # 多一条车道邻接信息
                        intersection_id=str(i)
                    )
                elif agent_name in ["Hmasac","MASACGraph","MAPPOGraph","Hmappo"]:
                    agent = DIC_AGENTS[agent_name](
                        dic_agent_conf=self.dic_agent_conf,
                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                        dic_path=self.dic_path,
                        cnt_round=self.cnt_round,
                        best_round=best_round,
                        intersection_id=str(i)
                    )
                else:              
                    agent = DIC_AGENTS[agent_name](
                        dic_agent_conf=self.dic_agent_conf,
                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                        dic_path=self.dic_path,
                        cnt_round=self.cnt_round, 
                        best_round=best_round,
                        intersection_id=str(i)
                    )

                self.agents[i] = agent  # 初始化每个agent内部的配置。
            print("Create intersection agent time: ", time.time()-start_time)


    # 主要是智能体choose_action，然后更新虚拟环境的state，动作默认最大只能执行step_num次。
    def generate(self):

        reset_env_start_time = time.time()
        done = False
        state = self.env.reset()  # 获取到虚拟环境的状态。
        step_num = 0
        reset_env_time = time.time() - reset_env_start_time  # 重置环境所耗费的时间

        running_start_time = time.time()

        # run_counts默认是3600，min_action_time默认是10或者1，step_num是执行动作的次数。
        while not done and step_num < int(self.dic_exp_conf["RUN_COUNTS"]/self.dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []  # 动作列表
            step_start_time = time.time()

            # 每个智能体根据当前状态选择一个动作，根据当前的step_num和当前状态。这里的choose_action的方法是在模型自身的py文件中
            for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):

                if self.dic_exp_conf["MODEL_NAME"] in ["CoLight", "GCN", "SimpleDQNOne","Hmasac","Hmappo","MASACGraph","MAPPOGraph"]:
                    one_state = state  # 这里的one_state的维度是（36，） {'cur_phase': [1], 'lane_num_vehicle': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'adjacency_matrix': [0, 5, 4, 1, 2], 'adjacency_matrix_lane': [[[-1, -1, -1, -1, -1], [21, 22, 23, -1, -1]], [[-1, -1, -1, -1, -1], [48, 49, 50, -1, -1]], [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]], [[57, 52, 56, -1, -1], [-1, -1, -1, -1, -1]], [[57, 52, 56, -1, -1], [-1, -1, -1, -1, -1]], [[57, 52, 56, -1, -1], [21, 22, 23, -1, -1]], [[14, 15, 19, -1, -1], [48, 49, 50, -1, -1]], [[14, 15, 19, -1, -1], [-1, -1, -1, -1, -1]], [[14, 15, 19, -1, -1], [-1, -1, -1, -1, -1]], [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]], [[-1, -1, -1, -1, -1], [21, 22, 23, -1, -1]], [[-1, -1, -1, -1, -1], [48, 49, 50, -1, -1]]]}
                    if self.dic_exp_conf["MODEL_NAME"] == 'CoLight':
                        action, _ = self.agents[i].choose_action(step_num, one_state)
                    elif self.dic_exp_conf["MODEL_NAME"] == 'GCN':
                        action = self.agents[i].choose_action(step_num, one_state)
                    elif self.dic_exp_conf["MODEL_NAME"] == "Hmasac":
                        action = self.agents[i].choose_action(step_num, one_state)
                    elif self.dic_exp_conf["MODEL_NAME"] == "Hmappo":
                        action = self.agents[i].choose_action(step_num, one_state)
                    elif self.dic_exp_conf["MODEL_NAME"] == "MASACGraph":
                        action = self.agents[i].choose_action(step_num, one_state)
                    elif self.dic_exp_conf["MODEL_NAME"] == "MAPPOGraph":
                        action = self.agents[i].choose_action(step_num, one_state)
                        pass
                    else: # simpleDQNOne
                        if True:
                            action = self.agents[i].choose_action(step_num, one_state)
                        else:
                            action = self.agents[i].choose_action_separate(step_num, one_state)
                    action_list = action
                else:
                    one_state = state[i]
                    action = self.agents[i].choose_action(step_num, one_state)
                    action_list.append(action)

            # 通过动作得到下一状态、即时奖励和是否达到终点（即智能体不需要/无法再执行动作）
            next_state, reward, done, _ = self.env.step(action_list)
            # 这里的running_time是执行完动作后所花费的时间。
            print("time: {0}, running_time: {1}".format(self.env.get_current_time()-self.dic_traffic_env_conf["MIN_ACTION_TIME"],
                                                        time.time()-step_start_time))
            state = next_state
            step_num += 1

        # 这里的running_time是执行完所有的step_num之后所花费的时间
        running_time = time.time() - running_start_time

        log_start_time = time.time()
        print("start logging")
        self.env.bulk_log_multi_process()  # 多进程方式将样本信息进行存储，以便放到construct_samples中进行构建样本。
        log_time = time.time() - log_start_time

        self.env.end_sumo()
        print("reset_env_time: ", reset_env_time)
        print("running_time: ", running_time)  # 执行完所有的step_num之后所花费的时间
        print("log_time: ", log_time)
