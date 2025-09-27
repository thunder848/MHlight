import json
import os
import shutil
import xml.etree.ElementTree as ET
from typing import BinaryIO

from generator import Generator
from construct_sample import ConstructSample
from updater import Updater
from multiprocessing import Process, Pool
from model_pool import ModelPool
import random
import pickle
import model_test
import pandas as pd
import numpy as np
from math import isnan
import sys
import time
import traceback

# 主要工作内容就是包装，
# 首先将其他几个板块封装在管道内，然后启动模拟器环境，运行一段时间，从原始日志数据构建样本，更新模型和模型池
class Pipeline:
    _LIST_SUMO_FILES = [
        "cross.tll.xml",
        "cross.car.type.xml",
        "cross.con.xml",
        "cross.edg.xml",
        "cross.net.xml",
        "cross.netccfg",
        "cross.nod.xml",
        "cross.sumocfg",
        "cross.typ.xml"
    ]

    # 关于下面的和这个@staticmethod，是指该方法是一个静态方法，优点是节省开销。下面的这个方法，因为不使用sumo环境，所以...
    @staticmethod
    def _set_traffic_file(sumo_config_file_tmp_name, sumo_config_file_output_name, list_traffic_file_name):

        # update sumocfg，更新sumo的配置
        sumo_cfg = ET.parse(sumo_config_file_tmp_name)
        config_node = sumo_cfg.getroot()
        input_node = config_node.find("input")
        for route_files in input_node.findall("route-files"):
            input_node.remove(route_files)
        input_node.append(
            ET.Element("route-files", attrib={"value": ",".join(list_traffic_file_name)}))
        sumo_cfg.write(sumo_config_file_output_name)

    # 专门检查几个目录是否出现错误
    def _path_check(self):
        # check path
        if os.path.exists(self.dic_path["PATH_TO_WORK_DIRECTORY"]):  # 是有这个工作目录在的
            if self.dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":  # 这里的这个default是config.py文件内的
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(self.dic_path["PATH_TO_WORK_DIRECTORY"]) # 当前工作目录 records

        if os.path.exists(self.dic_path["PATH_TO_MODEL"]):
            if self.dic_path["PATH_TO_MODEL"] != "model/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(self.dic_path["PATH_TO_MODEL"]) # model

        if os.path.exists(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"]): # records/initial
            pass
        else:
            os.makedirs(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"])

        if os.path.exists(self.dic_path["PATH_TO_PRETRAIN_MODEL"]): # model/initial
            pass
        else:
            os.makedirs(self.dic_path["PATH_TO_PRETRAIN_MODEL"])

    # 将三个配置文件写到records内的文件目录下。
    def _copy_conf_file(self, path=None):
        # write conf files
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]  # 将records内的文件目录路径，复制给path
        json.dump(self.dic_exp_conf, open(os.path.join(path, "exp.conf"), "w"),
                  indent=4)
        json.dump(self.dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"),
                  indent=4)
        json.dump(self.dic_traffic_env_conf,
                  open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)

    # 复制sumo文件，因为不用到sumo文件，所以直接省略...
    def _copy_sumo_file(self, path=None):
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # copy sumo files
        for file_name in self._LIST_SUMO_FILES:
            shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], file_name),
                        os.path.join(path, file_name))
        for file_name in self.dic_exp_conf["TRAFFIC_FILE"]:
            shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], file_name),
                        os.path.join(path, file_name))

    # 简单的讲dic_path中的work_directory复制给path
    def _copy_anon_file(self, path=None):
        # hard code !!!
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # copy sumo files

        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["TRAFFIC_FILE"][0]),
                        os.path.join(path, self.dic_exp_conf["TRAFFIC_FILE"][0]))
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["ROADNET_FILE"]),
                    os.path.join(path, self.dic_exp_conf["ROADNET_FILE"]))

    # 调整sumo的环境，省略
    def _modify_sumo_file(self, path=None):
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # modify sumo files
        self._set_traffic_file(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "cross.sumocfg"),
                               os.path.join(path, "cross.sumocfg"),
                               self.dic_exp_conf["TRAFFIC_FILE"])

    # 该类的初始化，是传入三个配置字典以及一个路径字典
    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):

        # load configurations，载入布局文件
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        # do file operations
        self._path_check() # 检查records和model文件
        self._copy_conf_file() # 将exp.conf，agent.conf，traffic_env.conf复制到records目录下
        if self.dic_traffic_env_conf["SIMULATOR_TYPE"] == 'sumo':
            self._copy_sumo_file()
            self._modify_sumo_file()
        elif self.dic_traffic_env_conf["SIMULATOR_TYPE"] == 'anon':
            self._copy_anon_file() # 将车流和路网数据集复制到records目录下
        # test_duration
        self.test_duration = []  # 测试的持续时间，也就是最后的结果

        # 这里的sample_num是样本数量？，样本数量根据交叉口数量来制定，如果交叉口数量超过10，那么样本数量定为10，否则该是几个就是几个。
        sample_num = 10 if self.dic_traffic_env_conf["NUM_INTERSECTIONS"]>=10 else min(self.dic_traffic_env_conf["NUM_INTERSECTIONS"], 9)
        print("sample_num for early stopping:", sample_num)
        # 随机从交叉路口中提取几个样本， 超过十个的只取十个。
        self.sample_inter_id = random.sample(range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]), sample_num)

    # 提前停止测试，根据最后的15个数据，如果满足一定的要求（一组（15个）测试结果中，标准差/平均值<0.1且该组中最大测试结果<1.5平均值）就返回1，否则返回0。cnt_round表示当前训练的轮数
    def early_stopping(self, dic_path, cnt_round): # Todo multi-process
        print("decide whether to stop")
        early_stopping_start_time = time.time()
        record_dir = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", "round_"+str(cnt_round))  # 记录测试的目录，其中cnt_round是100

        ave_duration_all = []  # 使用列表记录抽取的样本路口的车辆的平均等待时间
        # compute duration，计算等待时间
        for inter_id in self.sample_inter_id:  # 对于每个交叉路口的样本
            try:
                df_vehicle_inter_0 = pd.read_csv(os.path.join(record_dir, "vehicle_inter_{0}.csv".format(inter_id)),
                                                 sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                 names=["vehicle_id", "enter_time", "leave_time"])  # vehicle_id 是flow_x_x
                duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values  # 所有车辆的持续时间
                ave_duration = np.mean([dur_time for dur_time in duration if not isnan(dur_time)])  # 求所有能正常驶出路网的车辆的平均时间，在一定时间内，不能正常驶出路口的时间为nan。
                ave_duration_all.append(ave_duration)
            except FileNotFoundError:
                error_dir = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")  # 报错目录
                if not os.path.exists(error_dir):
                    os.makedirs(error_dir)
                f = open(os.path.join(error_dir, "error_info.txt"), "a")
                f.write("Fail to read csv of inter {0} in early stopping of round {1}\n".format(inter_id, cnt_round))
                f.close()
                pass

        ave_duration = np.mean(ave_duration_all)  # 计算所抽取的所有样本路口的所有车辆的平均等待时间
        self.test_duration.append(ave_duration)
        early_stopping_end_time = time.time()  # TODO(solved) 这里暂时有个bug，但是暂时不会使用到这个函数，bug已经解决，主要是在165行代码又出现了一个time，将其更改为dur_time之后，bug解决。
        print("early_stopping time: {0}".format(early_stopping_end_time - early_stopping_start_time))  # 这里主要是计算所有车辆的在全部样本路口的平均等待时间所花费的时间
        if len(self.test_duration) < 30:  # 当test_duration的长度小于30时，也就是看执行early_stopping执行了多少次。
            return 0
        else:
            duration_under_exam = np.array(self.test_duration[-15:])  # 取出末尾的15次测试时间，存为array，当test_duration超过29时，每执行一次early_stopping，都会取出最后的15个平均等待时间
            mean_duration = np.mean(duration_under_exam)
            std_duration = np.std(duration_under_exam)
            max_duration = np.max(duration_under_exam)  # 找里面耗时最长的测试时间。
            if std_duration/mean_duration < 0.1 and max_duration < 1.5 * mean_duration:  # 这样设置是为了？
                return 1
            else:
                return 0

    # 调用generator文件内的方法。包括下面的两个wrapper都是封装好的，可以直接使用possess或者单独调用。
    def generator_wrapper(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, best_round=None):
        generator = Generator(cnt_round=cnt_round,
                              cnt_gen=cnt_gen,
                              dic_path=dic_path,
                              dic_exp_conf=dic_exp_conf,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf,
                              best_round=best_round
                              )  # 创建生成器（生成环境）这个类
        print("make generator")
        generator.generate()  # 启动生成器
        print("generator_wrapper end")
        return

    # 调用update文件内的方法。
    def updater_wrapper(self, cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, best_round=None, bar_round=None):

        updater = Updater(
            cnt_round=cnt_round,
            dic_agent_conf=dic_agent_conf,
            dic_exp_conf=dic_exp_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            best_round=best_round,
            bar_round=bar_round
        ) 

        updater.load_sample_for_agents()  # 为智能体加载样本
        updater.update_network_for_agents()  # 为智能体更新网络
        print("updater_wrapper end")
        return

    # 调用model_pool文件内的方法。模型池的封装
    def model_pool_wrapper(self, dic_path, dic_exp_conf, cnt_round):
        model_pool = ModelPool(dic_path, dic_exp_conf)
        model_pool.model_compare(cnt_round)
        model_pool.dump_model_pool()


        return
        #self.best_round = model_pool.get()
        #print("self.best_round", self.best_round)

    # 相当于对样本进行下采样，从原始日志文件中进行采样，提取部分数据，采样规则是每隔10条取一条，生成新的采样后的日志文件，TODO：可能存在问题
    def downsample(self, path_to_log, i):
        # 其中path_to_log是在records中的gengerator目录路径。
        path_to_pkl = os.path.join(path_to_log, "inter_{0}.pkl".format(i))  # 目前的i是36,也就是可能是6x6得到的。
        f_logging_data: BinaryIO  # 声明f_logging_data是一个可以进行二进制IO的文件对象
        with open(path_to_pkl, "rb") as f_logging_data:  # 使用二进制的方式打开path_to_pkl路径下的文件，使f_logging_data作为这个文件的文件对象
            try:
                logging_data = pickle.load(f_logging_data)  # 从日志文件对象中加载数据到data
                subset_data = logging_data[::10]  # 从前往后采样出每隔10条采集一条数据
                #print(subset_data)
                os.remove(path_to_pkl)  # 删除原始pkl文件
                with open(path_to_pkl, "wb") as f_subset:  # 打开pkl文件用于写入
                    try:
                        pickle.dump(subset_data, f_subset)  # 将采集到的数据写入f_subset文件
                    except Exception as e:
                        print("----------------------------")
                        print("Error occurs when WRITING pickles when down sampling for inter {0}".format(i))
                        print('traceback.format_exc():\n%s' % traceback.format_exc())
                        print("----------------------------")

            except Exception as e:
                # print("CANNOT READ %s"%path_to_pkl)
                print("----------------------------")
                print("Error occurs when READING pickles when down sampling for inter {0}, {1}".format(i, f_logging_data))
                print('traceback.format_exc():\n%s' % traceback.format_exc())
                print("----------------------------")

    # 为每个交叉路口加载样本，这里的i是交叉路口的索引号，范围是0~35
    def downsample_for_system(self, path_to_log, dic_traffic_env_conf):
        for i in range(dic_traffic_env_conf['NUM_INTERSECTIONS']):  # 按理说i的值就是0~35
            self.downsample(path_to_log, i)

    # 调用construct_sample文件内的方法。主要是通过多进程来加速样本构建
    def construct_sample_multi_process(self, train_round, cnt_round, batch_size=200):  # batch_size默认是200。
        cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                             dic_traffic_env_conf=self.dic_traffic_env_conf)
        if batch_size > self.dic_traffic_env_conf['NUM_INTERSECTIONS']:
            batch_size_run = self.dic_traffic_env_conf['NUM_INTERSECTIONS']  # batch_size_run变为交叉口大小。
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, self.dic_traffic_env_conf['NUM_INTERSECTIONS'], batch_size_run):
            start = batch
            stop = min(batch + batch_size, self.dic_traffic_env_conf['NUM_INTERSECTIONS'])  # 其实这里使用batch_size不如使用run，但是实际没区别。
            process_list.append(Process(target=self.construct_sample_batch, args=(cs, start, stop)))  # 多进程构建样本

        # 依次启动进程列表里的所有进程
        for t in process_list:
            t.start()
        # 依次加入，等待所有子进程完成后，才执行主线程。
        for t in process_list:
            t.join()

    # 这个函数用来计算start到stop这个batch范围内的交叉口中，每个交叉所得到的即时奖励。
    def construct_sample_batch(self, cs, start, stop):
        for inter_id in range(start, stop):
            print("make construct_sample_wrapper for ", inter_id)
            cs.make_reward(inter_id)
        
    # pipeline文件的运行文件
    def run(self, multi_process=False):

        best_round, bar_round = None, None  # 记录最优回合和采样回合

        # 完成记录训练文件的初始化，即完成表头工作，几个时间可以从running_time中看到。
        f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "w")  # 在records下打开running_time文件，用于记录训练过程时间
        f_time.write("generator_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")  # 写入表头文件
        f_time.close()

        # 如果实验配置环境中有预训练，由于默认都是false
        if self.dic_exp_conf["PRETRAIN"]:
            if os.listdir(self.dic_path["PATH_TO_PRETRAIN_MODEL"]):  # 该文件的路径在model的initial中。
                for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):
                    #TODO:only suitable for CoLight
                    shutil.copy(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                            "round_0_inter_%d.h5" % i),
                                os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_%d.h5"%i))  # 将预训练模型的round_0文件的内容复制到模型文件中。
            else:  # 如果没有预处理模型的文件路径
                if not os.listdir(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"]):  # 如果也没有预处理模型的工作目录
                    for cnt_round in range(self.dic_exp_conf["PRETRAIN_NUM_ROUNDS"]):  # 开始预训练
                        print("round %d starts" % cnt_round)  # cnt_round表示当前预训练的轮数

                        process_list = []

                        # ==============  先形成generator：其作用是载入模型，然后启动模拟器环境 =============
                        if multi_process:  # 如果多进程处理
                            for cnt_gen in range(self.dic_exp_conf["PRETRAIN_NUM_GENERATORS"]):
                                p = Process(target=self.generator_wrapper,
                                            args=(cnt_round, cnt_gen, self.dic_path, self.dic_exp_conf,
                                                  self.dic_agent_conf, self.dic_traffic_env_conf, best_round)
                                            )  # 其中cnt_gen是当前生成器的编号
                                print("before")
                                p.start()
                                print("end")
                                process_list.append(p)
                            print("before join")
                            for p in process_list:
                                p.join()
                            print("end join")
                        else:  # 如果没有多进程，则根据预训练的生成器数量循环去进行generator_wrapper。
                            for cnt_gen in range(self.dic_exp_conf["PRETRAIN_NUM_GENERATORS"]):  # pretrain_num_generators默认是10/15。
                                self.generator_wrapper(cnt_round=cnt_round,
                                                       cnt_gen=cnt_gen,
                                                       dic_path=self.dic_path,
                                                       dic_exp_conf=self.dic_exp_conf,
                                                       dic_agent_conf=self.dic_agent_conf,
                                                       dic_traffic_env_conf=self.dic_traffic_env_conf,
                                                       best_round=best_round)

                        # ==============  make samples，生成样本并计算即时奖励 =============
                        # make samples and determine which samples are good

                        train_round = os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"], "train_round")  # 预训练的文件路径
                        if not os.path.exists(train_round):
                            os.makedirs(train_round)
                        cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                                             dic_traffic_env_conf=self.dic_traffic_env_conf)  # 调用construct_sample文件内的方法构建样本
                        cs.make_reward()  # 计算样本的及时奖励

                # ==============  update，模型更新 =============
                if self.dic_exp_conf["MODEL_NAME"] in self.dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:  # 查看此前的模型是否在需要更新的模型列表中，模型列表可以从config文件中找到
                    if multi_process:
                        p = Process(target=self.updater_wrapper,  # 更新模块已经封装在updater_Wrapper中
                                    args=(0,
                                          self.dic_agent_conf,
                                          self.dic_exp_conf,
                                          self.dic_traffic_env_conf,
                                          self.dic_path,
                                          best_round))
                        p.start()
                        p.join()
                    else:
                        self.updater_wrapper(cnt_round=0,
                                             dic_agent_conf=self.dic_agent_conf,
                                             dic_exp_conf=self.dic_exp_conf,
                                             dic_traffic_env_conf=self.dic_traffic_env_conf,
                                             dic_path=self.dic_path,
                                             best_round=best_round)

        # train with aggregate samples 使用全部样本一起进行训练
        if self.dic_exp_conf["AGGREGATE"]:
            if "aggregate.h5" in os.listdir("model/initial"):  # 如果aggregate.h5文件在该目录下
                shutil.copy("model/initial/aggregate.h5",
                            os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0.h5"))
            else:
                if multi_process:
                    p = Process(target=self.updater_wrapper,
                                args=(0,
                                      self.dic_agent_conf,
                                      self.dic_exp_conf,
                                      self.dic_traffic_env_conf,
                                      self.dic_path,
                                      best_round))
                    p.start()
                    p.join()
                else:
                    self.updater_wrapper(cnt_round=0,
                                         dic_agent_conf=self.dic_agent_conf,
                                         dic_exp_conf=self.dic_exp_conf,
                                         dic_traffic_env_conf=self.dic_traffic_env_conf,
                                         dic_path=self.dic_path,
                                         best_round=best_round)

        self.dic_exp_conf["PRETRAIN"] = False
        self.dic_exp_conf["AGGREGATE"] = False

        # trainf，cnt_round表示当前训练的轮数
        for cnt_round in range(self.dic_exp_conf["NUM_ROUNDS"]):  # 训练轮次100
            print("round %d starts" % cnt_round)

            round_start_time = time.time()  # 记录每轮开始时间

            process_list = []

            # 先载入模型，并启动模拟器环境，会得到next_state, reward, action等相关信息。
            print("==============  generator =============")
            generator_start_time = time.time()  # 记录生成器的开始时间
            if multi_process:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):  # 这里的生成器数量到底在实际中表现的是啥？值为4
                    p = Process(target=self.generator_wrapper,
                                args=(cnt_round, cnt_gen, self.dic_path, self.dic_exp_conf,
                                      self.dic_agent_conf, self.dic_traffic_env_conf, best_round)
                                )
                    print("before p")
                    p.start() #跳转到generator_wrapper
                    print("end p")
                    process_list.append(p)
                print("before join")
                for i in range(len(process_list)):
                    p = process_list[i]
                    print("generator %d to join" % i)
                    p.join()
                    print("generator %d finish join" % i)
                print("end join")
            else:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    self.generator_wrapper(cnt_round=cnt_round,
                                           cnt_gen=cnt_gen,
                                           dic_path=self.dic_path,
                                           dic_exp_conf=self.dic_exp_conf,
                                           dic_agent_conf=self.dic_agent_conf,
                                           dic_traffic_env_conf=self.dic_traffic_env_conf,
                                           best_round=best_round)
            generator_end_time = time.time()
            generator_total_time = generator_end_time - generator_start_time  # 启动生成器的总耗费时间

            # 创建样本，整个样本的创建都被封装在constructsample文件内，计算及时奖励也在那个文件内。
            print("==============  make samples =============")
            # make samples and determine which samples are good
            making_samples_start_time = time.time()

            train_round = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
            if not os.path.exists(train_round):
                os.makedirs(train_round)

            cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                                 dic_traffic_env_conf=self.dic_traffic_env_conf)
            cs.make_reward_for_system()

            # EvaluateSample()，计算创建样本所花费时间
            making_samples_end_time = time.time()
            making_samples_total_time = making_samples_end_time - making_samples_start_time

            # 更新网络，因为有的模型不需要进行更新，TODO(solved)：为啥会在模型更新后对样本进行下采样？，因为为了去测试模型是否达到要求从而提早停止训练。
            # 只有通过上述计算出reward之后，才会得到state，action，next_state，reward等这些信息。才会进行网络更新。但是网络更新函数是在agent内部，需要自己写。
            print("==============  update network =============")
            update_network_start_time = time.time()
            if self.dic_exp_conf["MODEL_NAME"] in self.dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
                if multi_process:
                    p = Process(target=self.updater_wrapper,
                                args=(cnt_round,
                                      self.dic_agent_conf,
                                      self.dic_exp_conf,
                                      self.dic_traffic_env_conf,
                                      self.dic_path,
                                      best_round,
                                      bar_round))
                    p.start()
                    print("update to join")
                    p.join()
                    print("update finish join")
                else:
                    self.updater_wrapper(cnt_round=cnt_round,
                                         dic_agent_conf=self.dic_agent_conf,
                                         dic_exp_conf=self.dic_exp_conf,
                                         dic_traffic_env_conf=self.dic_traffic_env_conf,
                                         dic_path=self.dic_path,
                                         best_round=best_round,
                                         bar_round=bar_round)
            # 这里进行下采样的操作，下采样是为了得到测试样本。
            if not self.dic_exp_conf["DEBUG"]:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                               "round_" + str(cnt_round), "generator_" + str(cnt_gen))  # 也在records中，是最底层的目录，在generator_str中，目前的cnt_gen的大小是4
                    try:
                        self.downsample_for_system(path_to_log, self.dic_traffic_env_conf)  # 对样本进行下采样，也就是提取出部分样本，还是记录到原始样本文件中
                    except Exception as e:
                        print("----------------------------")
                        print("Error occurs when downsampling for round {0} generator {1}".format(cnt_round, cnt_gen))
                        print("traceback.format_exc():\n%s" % traceback.format_exc())
                        print("----------------------------")
            update_network_end_time = time.time()
            update_network_total_time = update_network_end_time - update_network_start_time

            # 测试评估，使用的是model_test文件内的方法，这里应该会得到在当前已经完成模型更新的前提下车辆流量经过路口的平均等待时间，并且保存在records的test_round下的vehicle_inter_x.csv文件中
            print("==============  test evaluation =============")

            test_evaluation_start_time = time.time()
            if multi_process:
                p = Process(target=model_test.test,
                            args=(self.dic_path["PATH_TO_MODEL"], cnt_round, self.dic_exp_conf["RUN_COUNTS"], self.dic_traffic_env_conf, False))
                p.start()
                if self.dic_exp_conf["EARLY_STOP"]:
                    p.join()
            else:
                model_test.test(self.dic_path["PATH_TO_MODEL"], cnt_round, self.dic_exp_conf["RUN_COUNTS"], self.dic_traffic_env_conf, if_gui=False)

            test_evaluation_end_time = time.time()
            test_evaluation_total_time = test_evaluation_end_time - test_evaluation_start_time

            # 提前停止，也就是此时训练的轮数已经达到了early_stopping的标准，则退出训练，由上一个测试评估产生的平均等待时间来完成下面的是否提前停止训练。
            print('==============  early stopping =============')
            if self.dic_exp_conf["EARLY_STOP"]:
                flag = self.early_stopping(self.dic_path, cnt_round)
                if flag == 1:
                    print("early stopping!")
                    print("training ends at round %s" % cnt_round)
                    break  # 因为已经满足了此时的测试要求，

            # 模型库评估，模型库是指保存一组训练好的模型，而不是简单的在原模型的基础上直接更新，对于好的模型，会保存下来。
            print('==============  model pool evaluation =============')
            # 开启模型库
            if self.dic_exp_conf["MODEL_POOL"] and cnt_round > 50:
                if multi_process:
                    p = Process(target=self.model_pool_wrapper,
                                args=(self.dic_path,
                                      self.dic_exp_conf,
                                      cnt_round),
                                )
                    p.start()
                    print("model_pool to join")
                    p.join()
                    print("model_pool finish join")
                else:
                    self.model_pool_wrapper(dic_path=self.dic_path,
                                            dic_exp_conf=self.dic_exp_conf,
                                            cnt_round=cnt_round)
                model_pool_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "best_model.pkl")  # 暂时没找到这个文件，因为没有开启模型库
                if os.path.exists(model_pool_dir):
                    model_pool = pickle.load(open(model_pool_dir, "rb"))  # 从model_pool_dir文件中读取信息到model_pool
                    ind = random.randint(0, len(model_pool) - 1)
                    best_round = model_pool[ind][0]  # 得到训练的最佳轮次，
                    ind_bar = random.randint(0, len(model_pool) - 1)
                    flag = 0
                    while ind_bar == ind and flag < 10:
                        ind_bar = random.randint(0, len(model_pool) - 1)
                        flag += 1
                    # bar_round = model_pool[ind_bar][0]
                    bar_round = None
                else:
                    best_round = None
                    bar_round = None

                # downsample
                if not self.dic_exp_conf["DEBUG"]:
                    path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "test_round",
                                               "round_" + str(cnt_round))
                    self.downsample_for_system(path_to_log, self.dic_traffic_env_conf)
            # 没开启模型库
            else:
                best_round = None

            print("best_round: ", best_round)

            print("Generator time: ", generator_total_time)
            print("Making samples time:", making_samples_total_time)
            print("update_network time:", update_network_total_time)
            print("test_evaluation time:", test_evaluation_total_time)

            print("round {0} ends, total_time: {1}".format(cnt_round, time.time()-round_start_time))
            f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],"running_time.csv"),"a")  # 里面记载着上述的几个时间，然后写到running_time.csv文件中
            f_time.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(generator_total_time,making_samples_total_time,
                                                          update_network_total_time,test_evaluation_total_time,
                                                          time.time()-round_start_time))
            f_time.close()

            if cnt_round == 51:
                print()
                pass


