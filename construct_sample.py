import numpy as np
import pickle
from multiprocessing import Pool, Process
import os
import traceback
import pandas as pd

# 根据原始数据构建训练样本，在配置中选择所需的状态功能，然后计算。需要知道这个类到底做了哪些事？
class ConstructSample:
    # 构架样本类的初始化
    def __init__(self, path_to_samples, cnt_round, dic_traffic_env_conf):
        self.parent_dir = path_to_samples
        self.path_to_samples = path_to_samples + "/round_" + str(cnt_round)
        self.cnt_round = cnt_round
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.logging_data_list_per_gen = None
        self.hidden_states_list = None
        self.samples = []
        self.samples_all_intersection = [None]*self.dic_traffic_env_conf['NUM_INTERSECTIONS']

    # 加载交叉路口i的样本日志文件，然后返回1和logging_data，这里的folder文件是generator目录。
    def load_data(self, folder, i):
        try:
            f_logging_data = open(os.path.join(self.path_to_samples, folder, "inter_{0}.pkl".format(i)), "rb")
            logging_data = pickle.load(f_logging_data)  # 有的样本是可以正常执行try的，但是有的样本则只能exception。
            f_logging_data.close()
            return 1, logging_data

        except Exception as e:
            print("Error occurs when making samples for inter {0}".format(i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0, None

    # 实现所有交叉路口的样本加载，从folder文件中加载数据并保存到logging_data_list_per_gen中
    def load_data_for_system(self, folder):
        '''
        Load data for all intersections in one folder
        :param folder:
        :return: a list of logging data of one intersection for one folder
        '''
        self.logging_data_list_per_gen = []
        # load settings
        print("Load data for system in ", folder)
        self.measure_time = self.dic_traffic_env_conf["MEASURE_TIME"]
        self.interval = self.dic_traffic_env_conf["MIN_ACTION_TIME"]  # 在config文件中是10

        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            pass_code, logging_data = self.load_data(folder, i)  # 传入对应的交叉口i的样本日志文件，inter_x.pkl文件
            if pass_code == 0:
                return 0
            self.logging_data_list_per_gen.append(logging_data)
        return 1

    # 实现所有交叉路口的隐藏状态数据，TODO：隐藏状态是用来干啥的？给LSTM等需要一些旧数据的算法腾地？
    def load_hidden_state_for_system(self, folder):
        print("loading hidden states: {0}".format(os.path.join(self.path_to_samples, folder, "hidden_states.pkl")))
        # load settings
        if self.hidden_states_list is None:
            self.hidden_states_list = []

        try:
            f_hidden_state_data = open(os.path.join(self.path_to_samples, folder, "hidden_states.pkl"), "rb")
            hidden_state_data = pickle.load(f_hidden_state_data) # hidden state_data is a list of numpy array
            # print(hidden_state_data)
            print(len(hidden_state_data))
            hidden_state_data_h_c = np.stack(hidden_state_data, axis=2)
            hidden_state_data_h_c = pd.Series(list(hidden_state_data_h_c))
            next_hidden_state_data_h_c = hidden_state_data_h_c.shift(-1)
            hidden_state_data_h_c_with_next = pd.concat([hidden_state_data_h_c, next_hidden_state_data_h_c], axis=1)
            hidden_state_data_h_c_with_next.columns = ['cur_hidden','next_hidden']
            self.hidden_states_list.append(hidden_state_data_h_c_with_next[:-1].values)
            return 1
        except Exception as e:
            print("Error occurs when loading hidden states in ", folder)
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0

    # 构建状态，需要弄清楚features里面是啥，相当于就是将i交叉口的状态-当前相位这个映射存放在state_after_selection中。
    # 这里的time是0~3600
    def construct_state(self, features, time, i):
        '''
        :param features:
        :param time:
        :param i:  intersection id
        :return:
        '''
        state = self.logging_data_list_per_gen[i][time]  # 根据时间和路口索引得到当前状态
        assert time == state["time"]
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            state_after_selection = {}
            # 这里的value[0]or[1]是选择虚拟环境Cityflow还是sumo的，而phase则是类似WSES、WLRL等。
            for key, value in state["state"].items():
                if key in features:
                    if "cur_phase" in key:
                        state_after_selection[key] = self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']][value[0]]
                    else:
                        state_after_selection[key] = value
        else:
            state_after_selection = {key: value for key, value in state["state"].items() if key in features}
        # print(state_after_selection)
        return state_after_selection

    # 这个方法和上面的那个一样，只不过是形参多一个，且返回值多了一个交叉口的编号。
    def _construct_state_process(self, features, time, state, i):
        assert time == state["time"]
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            state_after_selection = {}
            for key, value in state["state"].items():
                if key in features:
                    if "cur_phase" in key:
                        state_after_selection[key] = self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']][value[0]]
                    else:
                        state_after_selection[key] = value
        else:
            state_after_selection = {key: value for key, value in state["state"].items() if key in features}
        return state_after_selection, i

    # 从特征中得到奖励，但是需要知道rs是啥？返回一个reward字典。内部包含所有车道的车队长度，等待时间，所有左转车辆。。。
    def get_reward_from_features(self, rs):
        reward = {}
        reward["sum_lane_queue_length"] = np.sum(rs["lane_queue_length"])
        reward["sum_lane_wait_time"] = np.sum(rs["lane_sum_waiting_time"])
        reward["sum_lane_num_vehicle_left"] = np.sum(rs["lane_num_vehicle_left"])
        reward["sum_duration_vehicle_left"] = np.sum(rs["lane_sum_duration_vehicle_left"])
        reward["sum_num_vehicle_been_stopped_thres01"] = np.sum(rs["lane_num_vehicle_been_stopped_thres01"])
        reward["sum_num_vehicle_been_stopped_thres1"] = np.sum(rs["lane_num_vehicle_been_stopped_thres1"])
        ##TODO pressure
        reward['pressure'] = np.sum(rs["pressure"])
        return reward


    # 计算奖励根据rewards_components中的权重和组成成分来进行计算r
    def cal_reward(self, rs, rewards_components):
        r = 0
        for component, weight in rewards_components.items():
            if weight == 0:
                continue
            if component not in rs.keys():
                continue
            if rs[component] is None:
                continue
            r += rs[component] * weight
        return r

    # 构造奖励，返回路口i的在时间time~time+measure_time时间段的平均奖励，和在time+measure_time的及时奖励
    def construct_reward(self,rewards_components, time, i):

        rs = self.logging_data_list_per_gen[i][time + self.measure_time - 1]  # 注意这里的measure_time默认值是10.
        assert time + self.measure_time - 1 == rs["time"]
        rs = self.get_reward_from_features(rs['state'])  # 调用get_reward_from_feature得到rs字典
        r_instant = self.cal_reward(rs, rewards_components)  # 计算路口i的即时奖励

        # average，计算time~time+measure_time这个时间段的路口i的及时奖励的平均值。
        list_r = []
        for t in range(time, time + self.measure_time):
            #print("t is ", t)
            rs = self.logging_data_list_per_gen[i][t]
            assert t == rs["time"]
            rs = self.get_reward_from_features(rs['state'])
            r = self.cal_reward(rs, rewards_components)
            list_r.append(r)
        r_average = np.average(list_r)

        return r_instant, r_average

    # 评判路口i在时间time时的动作，返回valueerror和logging_data_list_per_gen[i][time]['action']
    def judge_action(self, time, i):
        if self.logging_data_list_per_gen[i][time]['action'] == -1:
            raise ValueError
        else:
            return self.logging_data_list_per_gen[i][time]['action']

    # 将训练样本制作完成[state, action, next_state, reward_average, reward_instant, time, folder+"-"+"round_{0}".format(self.cnt_round)]
    def make_reward(self, folder, i):
        '''
        make reward for one folder and one intersection,
        add the samples of one intersection into the list.samples_all_intersection[i]
        :param i: intersection id
        :return:
        '''
        if self.samples_all_intersection[i] is None:
            self.samples_all_intersection[i] = []

        if i % 100 == 0:
            print("make reward for inter {0} in folder {1}".format(i, folder))

        list_samples = []

        try:
            total_time = int(self.logging_data_list_per_gen[i][-1]['time'] + 1)  # 总的时间
            # construct samples
            time_count = 0
            for time in range(0, total_time - self.measure_time + 1, self.interval):  # 每10s为一个step，正好和measure_time=10对上
                state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"], time, i)
                reward_instant, reward_average = self.construct_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"],
                                                                       time, i)
                action = self.judge_action(time, i)

                if time + self.interval == total_time:
                    next_state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                                                      time + self.interval - 1, i)

                else:
                    next_state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                                                      time + self.interval, i)
                sample = [state, action, next_state, reward_average, reward_instant, time,
                          folder+"-"+"round_{0}".format(self.cnt_round)]
                list_samples.append(sample)


            # list_samples = self.evaluate_sample(list_samples)
            self.samples_all_intersection[i].extend(list_samples)  # 所有交叉口的样本（带有reward)存入
            return 1
        except Exception as e:
            print("Error occurs when making rewards in generator {0} for intersection {1}".format(folder, i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0

    # 将所有路口的状态、奖励、动作与时间步关系进行封装，然后保存，然后这里出现了folder文件
    def make_reward_for_system(self):
        '''
        Iterate all the generator folders, and load all the logging data for all intersections for that folder
        At last, save all the logging data for that intersection [all the generators]
        :return:
        '''
        for folder in os.listdir(self.path_to_samples):
            print(folder)
            if "generator" not in folder:
                continue

            if not self.evaluate_sample(folder) or not self.load_data_for_system(folder):
                continue

            for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                pass_code = self.make_reward(folder, i)
                if pass_code == 0:
                    continue

        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            self.dump_sample(self.samples_all_intersection[i], "inter_{0}".format(i))  # 将每个路口的samples写入到inter文件内？

    # 导出隐藏状态，就是将hidden_states_list中的隐   藏状态以二进制格式写入pkl文件中
    def dump_hidden_states(self, folder):
        total_hidden_states = np.vstack(self.hidden_states_list)
        print("dump_hidden_states shape:",total_hidden_states.shape)
        if folder == "":
            with open(os.path.join(self.parent_dir, "total_hidden_states.pkl"), "ab+") as f:
                pickle.dump(total_hidden_states, f, -1)
        elif "inter" in folder:
            with open(os.path.join(self.parent_dir, "total_hidden_states_{0}.pkl".format(folder)), "ab+") as f:
                pickle.dump(total_hidden_states, f, -1)
        else:
            with open(os.path.join(self.path_to_samples, folder, "hidden_states_{0}.pkl".format(folder)), 'wb') as f:
                pickle.dump(total_hidden_states, f, -1)


    # def evaluate_sample(self,list_samples):
    #     return list_samples

    # 对样本进行评估，车辆数是否大于车流量volume*num_row,模拟的轮数是否超过40次。
    def evaluate_sample(self, generator_folder):
        return True
        print("Evaluate samples")
        list_files = os.listdir(os.path.join(self.path_to_samples, generator_folder, ""))
        df = []
        # print(list_files)
        for file in list_files:
            if ".csv" not in file:
                continue
            data = pd.read_csv(os.path.join(self.path_to_samples, generator_folder, file))  # 将vehicle_inter_xx.csv文件读入data中
            df.append(data)
        df = pd.concat(df)
        num_vehicles = len(df['Unnamed: 0'].unique()) - len(df[df['leave_time'].isna()]['leave_time'])
        if num_vehicles < self.dic_traffic_env_conf['VOLUME']* self.dic_traffic_env_conf['NUM_ROW'] and self.cnt_round > 40: # Todo Heuristic
            print("Dumpping samples from ", generator_folder)
            return False
        else:
            return True

    # 导出样本，将samples对象以二进制形式序列化，追加到folder的pkl文件中
    def dump_sample(self, samples, folder):
        if folder == "":
            with open(os.path.join(self.parent_dir, "total_samples.pkl"), "ab+") as f:
                pickle.dump(samples, f, -1)
        elif "inter" in folder:
            with open(os.path.join(self.parent_dir, "total_samples_{0}.pkl".format(folder)), "ab+") as f:
                pickle.dump(samples, f, -1)
        else:
            with open(os.path.join(self.path_to_samples, folder, "samples_{0}.pkl".format(folder)), 'wb') as f:
                pickle.dump(samples, f, -1)

#
# if __name__=="__main__":
#     # TODO:应该是这里的地址问题
#     path_to_samples = "/Users/Wingslet/PycharmProjects/RLSignal/records/test/anon_3_3_test/train_round"
#     generator_folder = "generator_0"
#
#     dic_traffic_env_conf  = {
#
#             "NUM_INTERSECTIONS": 9,
#             "ACTION_PATTERN": "set",
#             "MEASURE_TIME": 10,
#             "MIN_ACTION_TIME": 10,
#             "DEBUG": False,
#             "BINARY_PHASE_EXPANSION": True,
#             "FAST_COMPUTE": True,
#
#             "NEIGHBOR": False,
#             "MODEL_NAME": "STGAT",
#             "SIMULATOR_TYPE": "anon",
#
#
#
#             "SAVEREPLAY": False,
#             "NUM_ROW": 3,
#             "NUM_COL": 3,
#
#             "VOLUME": 300,
#             "ROADNET_FILE": "roadnet_{0}.json".format("3_3"),
#
#             "LIST_STATE_FEATURE": [
#                 "cur_phase",
#                 # "time_this_phase",
#                 # "vehicle_position_img",
#                 # "vehicle_speed_img",
#                 # "vehicle_acceleration_img",
#                 # "vehicle_waiting_time_img",
#                 "lane_num_vehicle",
#                 # "lane_num_vehicle_been_stopped_thres01",
#                 # "lane_num_vehicle_been_stopped_thres1",
#                 # "lane_queue_length",
#                 # "lane_num_vehicle_left",
#                 # "lane_sum_duration_vehicle_left",
#                 # "lane_sum_waiting_time",
#                 # "terminal",
#                 # "coming_vehicle",
#                 # "leaving_vehicle",
#                 # "pressure"
#
#                 # "adjacency_matrix",
#                 # "lane_queue_length",
#             ],
#
#                 "DIC_FEATURE_DIM": dict(
#                     D_LANE_QUEUE_LENGTH=(4,),
#                     D_LANE_NUM_VEHICLE=(4,),
#
#                     D_COMING_VEHICLE = (12,),
#                     D_LEAVING_VEHICLE = (12,),
#
#                     D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
#                     D_CUR_PHASE=(1,),
#                     D_NEXT_PHASE=(1,),
#                     D_TIME_THIS_PHASE=(1,),
#                     D_TERMINAL=(1,),
#                     D_LANE_SUM_WAITING_TIME=(4,),
#                     D_VEHICLE_POSITION_IMG=(4, 60,),
#                     D_VEHICLE_SPEED_IMG=(4, 60,),
#                     D_VEHICLE_WAITING_TIME_IMG=(4, 60,),
#
#                     D_PRESSURE=(1,),
#
#                     D_ADJACENCY_MATRIX=(2,),
#
#                 ),
#
#             "DIC_REWARD_INFO": {
#                 "flickering": 0,
#                 "sum_lane_queue_length": 0,
#                 "sum_lane_wait_time": 0,
#                 "sum_lane_num_vehicle_left": 0,
#                 "sum_duration_vehicle_left": 0,
#                 "sum_num_vehicle_been_stopped_thres01": 0,
#                 "sum_num_vehicle_been_stopped_thres1": -0.25,
#                 "pressure": 0  # -0.25
#             },
#
#             "LANE_NUM": {
#                 "LEFT": 1,
#                 "RIGHT": 1,
#                 "STRAIGHT": 1
#             },
#
#             "PHASE": {
#                 "sumo": {
#                     0: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
#                     1: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
#                     2: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
#                     3: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
#                 },
#                 "anon": {
#                     # 0: [0, 0, 0, 0, 0, 0, 0, 0],
#                     1: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
#                     2: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
#                     3: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
#                     4: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
#                     # 'WSWL',
#                     # 'ESEL',
#                     # 'WSES',
#                     # 'NSSS',
#                     # 'NSNL',
#                     # 'SSSL',
#                 },
#             }
#         }
#
#     cs = ConstructSample(path_to_samples, 0, dic_traffic_env_conf)
#     cs.make_reward_for_system()
#
