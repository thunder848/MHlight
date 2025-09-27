import json
import os
import pickle
from config import DIC_AGENTS, DIC_ENVS
from copy import deepcopy
import torch
import numpy as np

def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1


def downsample(path_to_log, i):
    path_to_pkl = os.path.join(path_to_log, "inter_{0}.pkl".format(i))
    with open(path_to_pkl, "rb") as f_logging_data:
        logging_data = pickle.load(f_logging_data)
    subset_data = logging_data[::10]
    os.remove(path_to_pkl)
    with open(path_to_pkl, "wb") as f_subset:
        pickle.dump(subset_data, f_subset)

def downsample_for_system(path_to_log,dic_traffic_env_conf):
    for i in range(dic_traffic_env_conf['NUM_INTERSECTIONS']):
        downsample(path_to_log,i)



# TODO test on multiple intersections
def test(model_dir, cnt_round, run_cnt, _dic_traffic_env_conf, if_gui):
    dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)
    records_dir = model_dir.replace("model", "records")  # 将原来的model/改成records/
    model_round = "round_%d"%cnt_round
    dic_path = {}
    dic_path["PATH_TO_MODEL"] = model_dir
    num_round = cnt_round
    dic_path["PATH_TO_WORK_DIRECTORY"] = records_dir

    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    with open(os.path.join(records_dir, "exp.conf"), "r") as f:
        dic_exp_conf = json.load(f)

    if os.path.exists(os.path.join(records_dir, "sumo_env.conf")):
        with open(os.path.join(records_dir, "sumo_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)
    elif os.path.exists(os.path.join(records_dir, "anon_env.conf")):
        with open(os.path.join(records_dir, "anon_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)

    dic_exp_conf["RUN_COUNTS"] = run_cnt
    dic_traffic_env_conf["IF_GUI"] = if_gui

    # dump dic_exp_conf
    with open(os.path.join(records_dir, "test_exp.conf"), "w") as f:
        json.dump(dic_exp_conf, f)


    if dic_exp_conf["MODEL_NAME"] in dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
        dic_agent_conf["EPSILON"] = 0  # dic_agent_conf["EPSILON"]  # + 0.1*cnt_gen
        dic_agent_conf["MIN_EPSILON"] = 0

    agents = []


    try:
        path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
        if not os.path.exists(path_to_log):
            os.makedirs(path_to_log)
        env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](path_to_log=path_to_log,
                                                               path_to_work_directory=dic_path[
                                                                   "PATH_TO_WORK_DIRECTORY"],
                                                               dic_traffic_env_conf=dic_traffic_env_conf)

        done = False
        state = env.reset()  # 得到36个路口的状态

        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = dic_exp_conf["MODEL_NAME"]
            if agent_name=='CoLight_Signal':
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=dic_agent_conf,
                    dic_traffic_env_conf=dic_traffic_env_conf,
                    dic_path=dic_path,
                    cnt_round=1,  # useless
                    inter_info=env.list_intersection,
                    intersection_id=str(i)
                )
            else:
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=dic_agent_conf,
                    dic_traffic_env_conf=dic_traffic_env_conf,
                    dic_path=dic_path,
                    cnt_round=1,  # useless
                    intersection_id=str(i)
                )
            agents.append(agent)

            
        # 将已经训练好的网络直接载入
        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            if dic_traffic_env_conf["ONE_MODEL"]:
                agents[i].load_network("{0}".format(model_round))
            else:
                agents[i].load_network(("{0}_inter_{1}".format(model_round, agents[i].intersection_id)))


        step_num = 0


        attention_dict = {}


        while not done and step_num < int(dic_exp_conf["RUN_COUNTS"] / dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []

            for i in range(dic_traffic_env_conf["NUM_AGENTS"]):

                if "CoLight" in dic_exp_conf["MODEL_NAME"]:
                    one_state = state  # dim(36,)
                    action_list, attention = agents[i].choose_action(step_num, one_state)  # 36个路口根据自己的状态全部选择一个动作。
                    cur_time = env.get_current_time()  # 此时虚拟环境的时间，会作为一个指标来进行评定
                    attention_dict[cur_time] = attention
                elif "Hmasac" or "MASACGraph" or "MAPPOGraph" or "Hmappo" in dic_exp_conf["MODEL_NAME"]:
                    one_state = state  # dim(36,)
#______________________________________________________________________________数据补齐
                    # intersection_dic = {
                    #     0: [1, 4],
                    #     1: [0, 2, 5],
                    #     2: [1, 3, 6],
                    #     3: [2, 7],
                    #     4: [0, 5, 8],
                    #     5: [1, 4, 6, 9],
                    #     6: [2, 5, 7, 10],
                    #     7: [3, 6, 11],
                    #     8: [4, 9, 12],
                    #     9: [5, 8, 10, 13],
                    #     10: [6, 9, 11, 14],
                    #     11: [7, 10, 15],
                    #     12: [8, 13],
                    #     13: [9, 12, 14],
                    #     14: [10, 13, 15],
                    #     15: [11, 14],
                    # }
                    # # random_indices = torch.tensor([6,1,13,4,8,11,15])
                    # random_indices = torch.tensor([6,1,13])
                    # for node_idx in random_indices:
                    #     # 获取该节点的邻居列表
                    #     neighbors = intersection_dic.get(int(node_idx), [])
                    #     neighbor_vehicle_counts = []
                    #     for neighbor_idx in neighbors:
                    #         # 确保邻居节点存在且有有效的lane_num_vehicle数据
                    #         if 0 <= neighbor_idx < len(one_state):
                    #             neighbor_data = one_state[neighbor_idx]
                    #             if 'lane_num_vehicle' in neighbor_data:
                    #                 neighbor_vehicle_counts.append(neighbor_data['lane_num_vehicle'])
                    #     # 转换为NumPy数组
                    #     arr = np.array(neighbor_vehicle_counts)
                    #
                    #     # 按列求和后取平均，并向上取整
                    #     avg_vehicle_counts = np.ceil(np.mean(arr, axis=0)).astype(int).tolist()
                    #     one_state[node_idx]['lane_num_vehicle'] = avg_vehicle_counts
#______________________________________________________________________________________________________
                    action_list = agents[i].choose_action(step_num, one_state)  # 36个路口根据自己的状态全部选择一个动作。
                    cur_time = env.get_current_time()  # 此时虚拟环境的时间，会作为一个指标来进行评定
                elif "GCN" in dic_exp_conf["MODEL_NAME"]:
                    one_state = state
                    # print('one_state:',one_state)
                    action_list = agents[i].choose_action(step_num, one_state)
                    # print('action_list:',action_list)
                elif "SimpleDQNOne" in dic_exp_conf["MODEL_NAME"]:
                    one_state = state
                    if True:
                        action_list = agents[i].choose_action(step_num, one_state)
                    else:
                        action_list = agents[i].choose_action_separate(step_num, one_state)
                else:
                    one_state = state[i]
                    action = agents[i].choose_action(step_num, one_state)
                    action_list.append(action)

            next_state, reward, done, _ = env.step(action_list)

            state = next_state
            step_num += 1
        # print('bulk_log_multi_process')
        env.bulk_log_multi_process()
        env.log_attention(attention_dict)
        env.end_sumo()
        if not dic_exp_conf["DEBUG"]:
            path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round",
                                       model_round)
            # print("downsample", path_to_log)
            downsample_for_system(path_to_log, dic_traffic_env_conf)
            # print("end down")

    except Exception as e:
        print(e)
        error_dir = model_dir.replace("model", "errors")
        if os.path.exists(error_dir):
            f = open(os.path.join(error_dir, "error_info.txt"), "a")
            f.write("round_%d fail to test model"%cnt_round)
            f.close()
        else:
            os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info.txt"), "a")
            f.write("round_%d fail to test model"%cnt_round)
            f.close()
