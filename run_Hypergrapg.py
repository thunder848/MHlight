import config
import copy
from pipeline import Pipeline
import os
import time
from multiprocessing import Process
import argparse
import os
import matplotlib
# matplotlib.use('TkAgg')

from script import get_traffic_volume

# 文件名后面的inter是交叉路口的索引
multi_process = True
TOP_K_ADJACENCY=-1
TOP_K_ADJACENCY_LANE=-1
PRETRAIN=False
NUM_ROUNDS=50  # 训练的轮数
EARLY_STOP=False
NEIGHBOR=False
SAVEREPLAY=False
ADJACENCY_BY_CONNECTION_OR_GEO=False
hangzhou_archive=True
ANON_PHASE_REPRE=[]


# 配置一系列参数，并返回
def parse_args():
    parser = argparse.ArgumentParser()
    # The file folder to create/log in
    parser.add_argument("--memo", type=str,
                        default='1105_afternoon_Hmasac_6_6_bi')  # 1_3,2_2,3_3,4_4, 这个文件在records下面，表示记录文件
    parser.add_argument("--env", type=int, default=1)  # env=1 means you will run CityFlow
    parser.add_argument("--gui", type=bool, default=False)
    parser.add_argument("--road_net", type=str, default='6_6')  # '1_2') # which road net you are going to run，路网结构
    parser.add_argument("--volume", type=str, default='300')  # '300'，表示车流量为300m/s，已经确定
    parser.add_argument("--suffix", type=str, default="0.3_bi")  # 0.3，表示上述memo文件的后缀

    global hangzhou_archive  # 定义下面的全局变量
    hangzhou_archive = False
    global TOP_K_ADJACENCY
    TOP_K_ADJACENCY = 5
    global TOP_K_ADJACENCY_LANE
    TOP_K_ADJACENCY_LANE = 5
    global NUM_ROUNDS
    NUM_ROUNDS = 100
    global EARLY_STOP
    EARLY_STOP = False
    global NEIGHBOR
    # TAKE CARE
    NEIGHBOR = False
    global SAVEREPLAY  # if you want to relay your simulation, set it to be True
    SAVEREPLAY = True
    global ADJACENCY_BY_CONNECTION_OR_GEO
    # TAKE CARE
    ADJACENCY_BY_CONNECTION_OR_GEO = False

    # modify:TOP_K_ADJACENCY in line 154
    global PRETRAIN
    PRETRAIN = False
    parser.add_argument("--mod", type=str, default='Hmasac')
    parser.add_argument("--cnt", type=int, default=3600)  # 3600,TODO:这里后序应该可以更改
    parser.add_argument("--gen", type=int, default=4)  # 4

    parser.add_argument("-all", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--onemodel", type=bool, default=False)

    parser.add_argument("--visible_gpu", type=str, default="-1")
    global ANON_PHASE_REPRE
    tt = parser.parse_args()
    if 'CoLight_Signal' in tt.mod:  # 如果在parser.parse_args().mod模型中，则使用12dim的，反之则使用8dim的
        # 12dim
        ANON_PHASE_REPRE = {
            # 0: [0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES', 这里的S是straight的意思，并且和右转弯合在一起了
            2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
            3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL', 这里的L是左转弯的意思，内部的10无实际意义
            4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
        }
    else:
        # 12dim
        ANON_PHASE_REPRE = {
            1: [0, 1, 0, 1, 0, 0, 0, 0],  # 下面的这个则是使用8dim去表示4个相位阶段。
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0]
        }
    print('agent_name:%s', tt.mod)
    print('ANON_PHASE_REPRE:', ANON_PHASE_REPRE)  # 当前匿名的相位阶段有四种，但是不清楚每种相位表示的含义。

    return parser.parse_args()  # 返回一系列参数


# 一些记录文件，将traffic_file_list内部交通流文件进行改名操作。
def memo_rename(traffic_file_list):
    new_name = ""
    for traffic_file in traffic_file_list:
        if "synthetic" in traffic_file:
            sta = traffic_file.rfind("-") + 1
            print(traffic_file, int(traffic_file[sta:-4]))
            new_name = new_name + "syn" + traffic_file[sta:-4] + "_"
        elif "cross" in traffic_file:
            sta = traffic_file.find("equal_") + len("equal_")
            end = traffic_file.find(".xml")
            new_name = new_name + "uniform" + traffic_file[sta:end] + "_"
        elif "flow" in traffic_file:
            new_name = traffic_file[:-4]
    new_name = new_name[:-1]
    return new_name

# 其功能是将两个字典进行融合，返回一个新的。
def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result

#  检查所有的work是否正常在工作，其中list_cur_p是一个线程列表，都正常则返回-1，有不正常的则返回索引。
def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1

#  管道包装器，就是将实验、强化学习、模拟器（交通环境）、记录日志这四者的配置进行包装，左后返回。
def pipeline_wrapper(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
    ppl = Pipeline(dic_exp_conf=dic_exp_conf, # experiment config
                   dic_agent_conf=dic_agent_conf, # RL agent config
                   dic_traffic_env_conf=dic_traffic_env_conf, # the simolation configuration
                   dic_path=dic_path # where should I save the logs?，就是一些记录应该保存到哪个地方
                   )
    global multi_process
    ppl.run(multi_process=multi_process)  # 将上述四部分进行管道包装，所谓的包装，主要是对配置进行包装。

    print("pipeline_wrapper end")  # 提示包装完毕
    return

#  主要的运行的文件
def main(memo, env, road_net, gui, volume, suffix, mod, cnt, gen, r_all, workers, onemodel):

    # main(args.memo, args.env, args.road_net, args.gui, args.volume, args.ratio, args.mod, args.cnt, args.gen)
    #Jinan_3_4
    NUM_COL = int(road_net.split('_')[0])  # 先获取路网列数
    NUM_ROW = int(road_net.split('_')[1])  # 获取路网行数
    num_intersections = NUM_ROW * NUM_COL  # 交叉口数量
    print('num_intersections:',num_intersections)

    ENVIRONMENT = ["sumo", "anon"][env]  # 看是否选择Cityflow还是sumo环境。env如果是1则选择anon也就是Cityflow，如果是0则是sumo环境。

    # 根据道路的长度来区分traffic_file_list内有多个traffic_file
    if r_all:
        traffic_file_list = [ENVIRONMENT+"_"+road_net+"_%d_%s" %(v,suffix) for v in range(100,400,100)]  # 100-400的路长，这个生成了多个交通文件，根据道路长度进行区分
    else:
        traffic_file_list=["{0}_{1}_{2}_{3}".format(ENVIRONMENT, road_net, volume, suffix)]  # 这里相当于构建一个traffic文件列表，每个文件名由format内部四个参数构成，并用_拼接

    # 如果是Cityflow环境，则在文件后面添加.json，如果是sumo则是加.xml
    if env:
        traffic_file_list = [i+ ".json" for i in traffic_file_list ]
    else:
        traffic_file_list = [i+ ".xml" for i in traffic_file_list ]

    process_list = []       # 多进程运行的文件列表
    n_workers = workers     #len(traffic_file_list) , workers是用来处理对应的traffic_file_list中的traffic_file文件。
    multi_process = True    # 是否展开多进程处理。

    global PRETRAIN
    global NUM_ROUNDS
    global EARLY_STOP
    # 解决掉上述的文件名之后，开始对每个交通文件内部的细节进行布置
    for traffic_file in traffic_file_list:  # 对于每个交通文件，都执行下列操作
        # 实验的配置字典，ps:这里的部署都与main输进来的有关，但是具体的deplpy需要和config.py文件进行merge
        dic_exp_conf_extra = {

            "RUN_COUNTS": cnt,  # 运行次数，默认是3600，是模拟器环境中用来控制step_num的
            "MODEL_NAME": mod,  # 模型名
            "TRAFFIC_FILE": [traffic_file], # here: change to multi_traffic，交通文件名

            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),  # 路网文件名

            "NUM_ROUNDS": NUM_ROUNDS,  # 训练的最大次数，默认是100
            "NUM_GENERATORS": gen,  # 生成器个数

            "MODEL_POOL": False,  # 是否开启模型库
            "NUM_BEST_MODEL": 3,  # 最好模型数

            "PRETRAIN": PRETRAIN,  # 是否加入预训练
            "PRETRAIN_MODEL_NAME":mod,  # 预训练模型
            "PRETRAIN_NUM_ROUNDS": 0,  # 预训练轮次
            "PRETRAIN_NUM_GENERATORS": 15,  # 预训练启动次数

            "AGGREGATE": False,  # 是否全部的样本一起进行训练
            "DEBUG": False,
            "EARLY_STOP": EARLY_STOP,
        }

        # 智能体的配置字典
        dic_agent_conf_extra = {
            "EPOCHS": 100,  # 智能体的迭代次数
            "SAMPLE_SIZE": 1000,  # 样本容量大小
            "MAX_MEMORY_LEN": 10000,  # 最大经验池大小？
            "UPDATE_Q_BAR_EVERY_C_ROUND": False,
            "UPDATE_Q_BAR_FREQ": 5,

            "N_LAYER": 2,  # 网络层数
            "TRAFFIC_FILE": traffic_file,
        }

        global TOP_K_ADJACENCY
        global TOP_K_ADJACENCY_LANE
        global NEIGHBOR
        global SAVEREPLAY
        global ADJACENCY_BY_CONNECTION_OR_GEO
        global ANON_PHASE_REPRE
        # 交通环境配置环境，最为复杂的一个
        dic_traffic_env_conf_extra = {
            "USE_LANE_ADJACENCY": True,  # 定义是否使用车道邻接关系
            "ONE_MODEL": onemodel,  # 是否使用一个模型控制所有交叉口，或者是每个路口都存在一个模型
            "NUM_AGENTS": num_intersections,  # 智能体数量就是交叉口数量
            "NUM_INTERSECTIONS": num_intersections,
            "ACTION_PATTERN": "set",  # 动作空间表示
            "MEASURE_TIME": 10,  # 测量时间
            "IF_GUI": gui,  # 是否图形化界面
            "DEBUG": False,
            "TOP_K_ADJACENCY": TOP_K_ADJACENCY,  # 默认值是5，车道邻接保留最近K个
            "ADJACENCY_BY_CONNECTION_OR_GEO": ADJACENCY_BY_CONNECTION_OR_GEO,  # 车道邻接根据拓扑或距离
            "TOP_K_ADJACENCY_LANE": TOP_K_ADJACENCY_LANE,  # 默认值是5，车道邻接保留最近的k条车道，上一个是k值，这个是车道
            "SIMULATOR_TYPE": ENVIRONMENT,  # 模拟器格式：对应的环境，可能是Cityflow or sumo
            "BINARY_PHASE_EXPANSION": True,  # 相位扩展
            "FAST_COMPUTE": True,

            "NEIGHBOR": NEIGHBOR,  # 是否有邻接节点
            "MODEL_NAME": mod,

            "SAVEREPLAY": SAVEREPLAY,  # 是否保存回放功能，这个可以在Cityflow框架中查看
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,

            "TRAFFIC_FILE": traffic_file,  # 车流量文件
            "VOLUME": volume,  # 道路长度
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            "phase_expansion": {  # 这边相位是如何扩展的？，反正从四个相位变成八个相位
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0],
                5: [1, 1, 0, 0, 0, 0, 0, 0],
                6: [0, 0, 1, 1, 0, 0, 0, 0],
                7: [0, 0, 0, 0, 0, 0, 1, 1],
                8: [0, 0, 0, 0, 1, 1, 0, 0]
            },

            "phase_expansion_4_lane": {  # 两条单向道，比如南->北，西->东,所以只用两个相位即可
                1: [1, 1, 0, 0],
                2: [0, 0, 1, 1],
            },

            "LIST_STATE_FEATURE": [  # 状态列表特征，涵盖一个当前相位和车辆数
                "cur_phase",
                "lane_num_vehicle",  # 每条车道上的车辆数
            ],

                "DIC_FEATURE_DIM": dict(  # 维度特征字典，也就是所有信息特征向量的维度是多少
                    D_LANE_QUEUE_LENGTH=(4,),  # 队列长度
                    D_LANE_NUM_VEHICLE=(4,),  # 驶向该路口的每个车道的车辆数

                    D_COMING_VEHICLE = (12,),  # 即将到来的车辆
                    D_LEAVING_VEHICLE = (12,),  # 即将离开的车辆

                    D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
                    D_CUR_PHASE=(1,),  # 当前相位
                    D_NEXT_PHASE=(1,),  # 下一个相位
                    D_TIME_THIS_PHASE=(1,),  # 这个相位的时间，比如30s或者因为延长导致的60s
                    D_TERMINAL=(1,),
                    D_LANE_SUM_WAITING_TIME=(4,),  # 所有等待路线等待时间总和。
                    D_VEHICLE_POSITION_IMG=(4, 60,),  # 车辆位置图片
                    D_VEHICLE_SPEED_IMG=(4, 60,),  # 车辆速度图片
                    D_VEHICLE_WAITING_TIME_IMG=(4, 60,), # 车辆等待图片

                    D_PRESSURE=(1,),

                    D_ADJACENCY_MATRIX=(2,),

                    D_ADJACENCY_MATRIX_LANE=(6,),

                ),

            "DIC_REWARD_INFO": {  # 奖励字典信息
                "flickering": 0,#-5,#
                "sum_lane_queue_length": 0,  # 车辆排队长度
                "sum_lane_wait_time": 0,  # 所有车辆的等待时间
                "sum_lane_num_vehicle_left": 0,#-1,#  ，左转车辆数
                "sum_duration_vehicle_left": 0,  # 左转车辆的持续时间
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": -0.25,
                "pressure": 0  # -0.25
            },

            "LANE_NUM": {  # 三种行驶方向
                "LEFT": 1,
                "RIGHT": 1,
                "STRAIGHT": 1
            },

            "PHASE": {  # 信号灯的one-hot编码
                "sumo": {
                    0: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                    1: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                    2: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                    3: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                },
                "anon":ANON_PHASE_REPRE,
            }
        }

        ## ==================== multi_phase ====================
        global hangzhou_archive
        if hangzhou_archive:
            template='Archive+2'
        elif volume=='jinan':
            template="Jinan"
        elif volume=='hangzhou':
            template='Hangzhou'
        elif volume=='newyork':
            template='NewYork'
        elif volume=='chacha':
            template='Chacha'
        elif volume=='dynamic_attention':
            template='dynamic_attention'
        elif dic_traffic_env_conf_extra["LANE_NUM"] == config._LS:  # 根据lane_num做出区分，其中LS表示该路网只支持左转和直行，S表示只支持直行
            template = "template_ls"
        elif dic_traffic_env_conf_extra["LANE_NUM"] == config._S:
            template = "template_s"
        elif dic_traffic_env_conf_extra["LANE_NUM"] == config._LSR:
            template = "template_lsr"
        else:
            raise ValueError

        # 根据neighbor在状态特征列表上添加具体特征以及对应特征的编号，0~3，相当于是把邻居的状态特征拼接到当前节点的状态特征上，等于扩充维度。
        if dic_traffic_env_conf_extra['NEIGHBOR']:
            list_feature = dic_traffic_env_conf_extra["LIST_STATE_FEATURE"].copy()
            for feature in list_feature:
                for i in range(4):
                    dic_traffic_env_conf_extra["LIST_STATE_FEATURE"].append(feature+"_"+str(i))

        # 如果模型在这三个之中
        if mod in ['CoLight','GCN','SimpleDQNOne','Hmasac','Hmappo']:
            dic_traffic_env_conf_extra["NUM_AGENTS"] = 1  # 智能体群只用一个即可
            dic_traffic_env_conf_extra['ONE_MODEL'] = False
            # 如果状态特征中没有邻接矩阵以及邻接矩阵的车道数，且模型也不是simpleDQNOne
            if "adjacency_matrix" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE'] and \
                "adjacency_matrix_lane" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE'] and \
                mod not in ['SimpleDQNOne']:
                dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("adjacency_matrix")
                dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("adjacency_matrix_lane")

                # 这一部分主要是解决邻接矩阵的拼接，如果ADJACENCY_BY_CONNECTION_OR_GEO是true则表示拼接，然后添加一系列特征，
                # 如果是false，则直接令D_ADJACENCY_MATRIX为5，即维度为5。
                if dic_traffic_env_conf_extra['ADJACENCY_BY_CONNECTION_OR_GEO']:
                    TOP_K_ADJACENCY = 5  # 应该是多注意力节点使用concat的方式
                    dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("connectivity")
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CONNECTIVITY'] = \
                        (5,)  # 由于创建了一个新的特征，所以设置该特征的维度为5
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX'] = \
                        (5,)   # 维度从2变成5
                else:
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX'] = \
                        (dic_traffic_env_conf_extra['TOP_K_ADJACENCY'],)  # TOP_K_ADJACENCY默认值是5

                if dic_traffic_env_conf_extra['USE_LANE_ADJACENCY']:
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX_LANE'] = \
                        (dic_traffic_env_conf_extra['TOP_K_ADJACENCY_LANE'],)  # 和TOP_K_ADJACENCY一样默认值是5

        # 如果模型不是上述三个，则将智能体数量设置为交叉路口数量
        else:
            dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]

        # 是否进行二进制的相位扩展，分为有邻居节点和无邻居节点两种
        if dic_traffic_env_conf_extra['BINARY_PHASE_EXPANSION']:
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE'] = (8,)  # 初始的维度是1维，首先扩展成8维二进制
            if dic_traffic_env_conf_extra['NEIGHBOR']:  # 是否有邻接节点，是，则在特征维度上继续添加周边四个相位的特征
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)  # 这个参数应该是驶向该路口的四条道路上的车辆数
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)

            # 如果没有邻接节点，则每个相位用1维表示，也就是说如果该节点是边缘节点，则他的相位对其不会产生影响。
            else:
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)


        print(traffic_file)
        prefix_intersections = str(road_net)  # 传进来的road_net，格式是：x_x。
        dic_path_extra = {  # 路径字典
            "PATH_TO_MODEL": os.path.join("model", memo, traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            # ptwd是anno....json...29那一级文件
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo, traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),

            "PATH_TO_DATA": os.path.join("data", template, prefix_intersections),  # 数据路径get
            "PATH_TO_PRETRAIN_MODEL": os.path.join("model", "initial", traffic_file),
            "PATH_TO_PRETRAIN_WORK_DIRECTORY": os.path.join("records", "initial", traffic_file),  # 这个路径暂时还没找到
            "PATH_TO_ERROR": os.path.join("errors", memo)
        }

        deploy_dic_exp_conf = merge(config.DIC_EXP_CONF, dic_exp_conf_extra)  # 这里的实验部署，因为是merge，所以某种程度上是实验的赋值
        deploy_dic_agent_conf = merge(getattr(config, "DIC_{0}_AGENT_CONF".format(mod.upper())),
                                      dic_agent_conf_extra)  # 将智能体的环境进行部署
        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)  # 方向是将本文件的更新到config的那个文件中。

        # TODO add agent_conf for different agents
        # deploy_dic_agent_conf_all = [deploy_dic_agent_conf for i in range(deploy_dic_traffic_env_conf["NUM_AGENTS"])]

        deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)  # 路径字典融合

        # 多进程处理，也就是在这里调用了pipeline中的封装技术，并且启动模拟环境，也就是dic_traffic_env_conf下的
        if multi_process:
            ppl = Process(target=pipeline_wrapper,
                          args=(deploy_dic_exp_conf,
                                deploy_dic_agent_conf,
                                deploy_dic_traffic_env_conf,
                                deploy_dic_path))
            process_list.append(ppl)
        else:
            pipeline_wrapper(dic_exp_conf=deploy_dic_exp_conf,
                             dic_agent_conf=deploy_dic_agent_conf,
                             dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                             dic_path=deploy_dic_path)

    if multi_process:
        for i in range(0, len(process_list), n_workers):
            i_max = min(len(process_list), i + n_workers)
            for j in range(i, i_max):
                print(j)
                print("start_traffic")
                process_list[j].start()
                print("after_traffic")
            for k in range(i, i_max):
                print("traffic to join", k)
                process_list[k].join()
                print("traffic finish join", k)


    return memo


if __name__ == "__main__":
    args = parse_args()
    #memo = "multi_phase/optimal_search_new/new_headway_anon"cd

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
    args.mod = "Hmasac"
    args.memo = 'HG_jinan_0.4'
    # args.memo = '1105_afternoon_Hmappo_4_4_real'
    args.road_net = '3_4'
    args.volume = 'jinan'
    args.suffix = 'real'
    #阈值为0.8
    main(args.memo, args.env, args.road_net, args.gui, args.volume,
         args.suffix, args.mod, args.cnt, args.gen, args.all, args.workers,
         args.onemodel)






