import copy
import time

from DDPG_network import Agent, Federated_Server, Federated_Server_AP, Actor, Critic
from robotic_env import RobotEnv
import random
import numpy as np
import tensorflow as tf

class Arguments():
    def __init__(self):
        # Federated Learning Settings
        self.N_robotics = 20
        self.N_APs=4
        self.local_epoch = 10
        self.episode=10
        self.move_num=10
        self.global_epoch=self.episode*self.move_num
        #参数
        self.LR_A = 0.0001  # learning rate for actor
        self.LR_C = 0.001  # learning rate for critic

        # Federated Unlearning Settings
        # 用于控制保存模型参数的轮数 1表示每轮保存一次的参数。
        self.unlearn_interval = 1  # Used to control how many rounds the model parameters are saved.1 represents the parameter saved once per round  N_itv in our paper.
        self.forget_client_idx = 2  # If want to forget, change None to the client index

        # If this parameter is set to False, only the global model after the final training is completed is output
        self.if_retrain = False  # If set to True, the global model is retrained using the FL-Retrain function, and data corresponding to the user for the forget_client_IDx number is discarded.

        self.if_unlearning = False  # If set to False, the global_train_once function will not skip users that need to be forgotten;If set to True, global_train_once skips the forgotten user during training
        # forget_local_epoch_ratio：当选择一个用户被遗忘时，其他用户需要在各自的数据集中进行几轮在线训练，得到模型收敛的大方向，从而提供模型收敛的大方向。
        self.forget_local_epoch_ratio = 0.5  # When a user is selected to be forgotten, other users need to train several rounds of on-line training in their respective data sets to obtain the general direction of model convergence in order to provide the general direction of model convergence.
        # forget_local_epoch_ratio*local_epoch Is the number of rounds of local training when we need to get the convergence direction of each local model
        # self.mia_oldGM = False


def unlearning(robotics,robot_movers,APs,old_GMs, old_CMs,FL_params):
    print("开始遗忘学习：")
    #判断
    if (FL_params.if_unlearning == False):
        raise ValueError('FL_params.if_unlearning should be set to True, if you want to unlearning with a certain user')

    if (not (FL_params.forget_client_idx in range(FL_params.N_robotics))):
        raise ValueError('FL_params.forget_client_idx is note assined correctly, forget_client_idx should in {}'.format(
            range(FL_params.N_robotics)))
    if (FL_params.unlearn_interval == 0 or FL_params.unlearn_interval > FL_params.global_epoch):
        raise ValueError(
            'FL_params.unlearn_interval should not be 0, or larger than the number of FL_params.global_epoch')
    numberOfRobots=FL_params.N_robotics-1
    numberOfAPs=FL_params.N_APs

    old_global_models = copy.deepcopy(old_GMs)
    old_client_models = copy.deepcopy(old_CMs)
    print(len(old_GMs))

    forget_client = FL_params.forget_client_idx
    #1、移除要忘记的机器人模型
    for ii in range(len(old_client_models)):
        temp = old_client_models[ii]
        temp.pop(forget_client)
        old_client_models.append(temp)
    #2、移除机器人
    robotics.pop(forget_client)
    robot_movers.pop(forget_client)
    sess_temp=tf.Session()
    actor1 = Actor(FL_params.LR_A, 8,'ta', [148], sess_temp,
                       32, 32, 1, batch_size=64, chkpt_dir='/checkpoint')
    critic1 = Critic(FL_params.LR_C, 8, 'tc', [148], sess_temp,
                         32, 32, chkpt_dir='/checkpoint')

    #3、重置经验池
    for ii in range(len(robotics)):
        robotics[ii].memory.clear()  #重置经验池
        robotics[ii].reset()
        #robotics[ii].actor.params=old_client_models[0][ii] #赋予第一次的值
    for ii in range(len(robot_movers)):
        robot_movers[ii].memory.clear()
        robotics[ii].reset()
    for ii in range(len(APs)):
        APs[ii].memory.clear()
        APs[ii].reset()

    #4、重置环境
    env = RobotEnv(numOfRobo=numberOfRobots, numOfAPs=numberOfAPs)  # number of robots and AP
    #设置要用的参数
    #遗忘后的全局模型
    unlearn_global_models = list()
    # 本地训练轮数
    # np.ceil向上取整
    FL_params.local_epoch = np.ceil(FL_params.local_epoch * FL_params.forget_local_epoch_ratio)
    FL_params.local_epoch = np.int16(FL_params.local_epoch)
    #训练用到的参数
    Server_robot = Federated_Server(numberOfRobots, name_actor='server_actor', name_critic='server_critic',
                                    input_dims=[148],
                                    n_actions=8, layer1_size=32, layer2_size=32)
    Server_AP = Federated_Server_AP(name_actor='server_actor_AP', name_critic='server_critic_AP', input_dims=[357],
                                    n_actions=8, layer1_size=32, layer2_size=32)
    score_Robot = 0
    score_AP = 0
    score1_history = []
    score0_history = []
    delay_history = []
    energy_history = []
    reject_history = []
    steps = 100
    Explore = 100000.
    warmed = 0
    new_state = 0
    epsilon = 1
    epsilon_move = 1
    all_client_params = list()  # 客户端的
    all_global_params = list()  # 全局的
    log_dir = "logs/"
    summary_writer = tf.summary.create_file_writer(log_dir)
    cnt = 0
    gn=0
    print('Local Calibration Training epoch = {}'.format(FL_params.local_epoch))
    print("开始近似训练：")
    for i in range(FL_params.episode):
        obs = env.reset()
        obs_move = obs
        a = 0
        test = 0  # this is for debugging, and when it is '1', we can choose action to debug
        for move in range(FL_params.move_num):
            Local_Observaion = []
            AP_observation = obs[numberOfRobots]  # the last observation is for access points
            for R in range(numberOfRobots):
                Local_Observaion.append(obs[R])
            epsilon_move -= 1 / Explore
            act_robot_moves = np.zeros((numberOfRobots, 2))
            # 移动是二维的
            if np.random.random() < epsilon_move:  # 贪婪探索
                for x in range(numberOfRobots):
                    act0_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act_robot_moves[x] = act0_robot_move
            else:
                for x in range(numberOfRobots):
                    act0_robot_move = robot_movers[x].choose_action(Local_Observaion[x])
                    act_robot_moves[x] = act0_robot_move

            # start federation loop
            for k in range(1):
                for j in range(FL_params.local_epoch):
                    # while not done:
                    Local_Observaion = []
                    AP_observation = obs[numberOfRobots]  # the last observation is for access points
                    for R in range(numberOfRobots):
                        Local_Observaion.append(obs[R])
                    epsilon -= 1 / Explore
                    a += 1
                    act_robots = np.zeros((numberOfRobots, 8))
                    act_APs = np.zeros((numberOfAPs, 8))
                    if np.random.random() <= epsilon:
                        for y in range(numberOfRobots):
                            act_robots[i] = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0,
                                                                                            1.0), np.random.uniform(
                                -1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0,
                                                                                            1.0), np.random.uniform(
                                -1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)

                        for y in range(numberOfAPs):
                            act_APs[y] = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(
                                -1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0,
                                                                                            1.0), np.random.uniform(
                                -1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)

                    else:
                        for y in range(numberOfRobots):
                            act_robots[i] = robotics[y].choose_action(Local_Observaion[y])
                        for y in range(numberOfAPs):
                            act_APs[y] = APs[y].choose_action(AP_observation)

                    new_state, reward_Robot, reward_AP, done, info, accept, AoI, energy, posX0, posY0, posX1, posY1 = env.step_task_offloading(
                        act_APs, act_robots, act_robot_moves)
                    for re in range(numberOfRobots):
                        robotics[re].remember(Local_Observaion[re], act_robots[re], reward_Robot[re], new_state[re],
                                              done)
                    for re in range(numberOfAPs):
                        APs[re].remember(AP_observation, act_APs[re], reward_AP[re], new_state[numberOfRobots], done)

                    for re in range(numberOfRobots):
                        robotics[re].learn()
                    for re in range(numberOfAPs):
                        APs[re].learn()

                    score_Robot += np.average(reward_Robot)
                    score_AP += np.average(reward_AP)
                    cnt = cnt + 1
                    tf.summary.scalar('Robots reward/forgotten', score_Robot, step=cnt)
                    tf.summary.scalar('APs reward/forgotten', score_AP, step=cnt)
                    print("forgotten Robots reward = ", score_Robot)
                    print("forgotten APs reward = ", score_AP)
                    if i % 10 == 0:
                        with open("03-accept_LR_high_action_corrected.txt", 'a') as reward_APs:
                            reward_APs.write(str(accept) + '\n')
                        with open("03-AoI_LR_high_action_corrected.txt", 'a') as AoI_file:
                            AoI_file.write(str(AoI) + '\n')
                        with open("03-energy_LR_high_action_corrected.txt", 'a') as energy_file:
                            energy_file.write(str(energy) + '\n')
                    obs = new_state
                    obs_move = new_state
                # calculate federation
                # start federation
                clients_params = list()  #
                for rn in range(numberOfRobots):
                    client0_params = robotics[rn].get_param()
                    clients_params.append(client0_params)
                    # 只有当这个的奖励超过平均的才会上传聚合
                    if reward_Robot[rn] >= np.average(reward_Robot):
                        Server_robot.actors_params[rn] = robotics[rn].get_param()
                        Server_robot.robot_sents[rn] = 1
                Server_AP.actor_paramAP1 = APs[0].get_param()
                Server_AP.actor_paramAP2 = APs[1].get_param()
                Server_AP.actor_paramAP3 = APs[2].get_param()
                Server_AP.actor_paramAP4 = APs[3].get_param()
                # 聚合
                actor_robot_new = Server_robot.federation()
                new_GM = unlearning_step_once(old_client_models[gn], clients_params, old_global_models[gn + 1],
                                              actor_robot_new)
                #actor_robot_new = Server_robot.federation()
                actor_AP_new = Server_AP.federation()
                gn=gn+1

                # 设置新的参数
                for f in range(numberOfRobots):
                    robotics[f].actor.params = new_GM
                for f in range(numberOfAPs):
                    APs[f].actor.params = actor_AP_new
                # 重新置为0
                for f in range(numberOfRobots):
                    Server_robot.robot_sents[f] = 0
                # end of federation

            with open("03-forgotten_reward_robot_LR_high_action_corrected.txt", 'a') as reward_robots:
                reward_robots.write(str(score_Robot) + '\n')
            with open("03-forgotten_reward_AP_LR_high_action_corrected.txt", 'a') as reward_APs:
                reward_APs.write(str(score_AP) + '\n')
            with open("03-forgotten_pos_0_high_action_corrected.txt", 'a') as pos0:
                pos0.write(str(posX0) + ', ' + str(posY0) + '\n')
            with open("03-forgotten_pos_1_high_action_corrected.txt", 'a') as pos1:
                pos1.write(str(posX1) + ', ' + str(posY1) + '\n')

            # 移动结束后，移动机器人也需要存储经验加学习
            for mr in range(numberOfRobots):
                robot_movers[mr].remember(Local_Observaion[mr], act_robot_moves[mr], score_Robot, new_state[mr], False)
            for mr in range(numberOfRobots):
                robot_movers[mr].learn()

            score_Robot = 0
            score_AP = 0



# 第k轮的旧的客户端模型，新的客户端模型，旧的全局模型，新的全局模型
def unlearning_step_once(old_client_models, new_client_models, global_model_before_forget, global_model_after_forget):
    old_param_update = dict()  # Model Params： oldCM - oldGM_t
    new_param_update = dict()  # Model Params： newCM - newGM_t

    new_global_model_state = global_model_after_forget  # newGM_t

    return_model_state = dict()  # newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||

    assert len(old_client_models) == len(new_client_models)

    for layer in global_model_before_forget.keys():
        old_param_update[layer] = 0 * global_model_before_forget[layer]
        new_param_update[layer] = 0 * global_model_before_forget[layer]

        return_model_state[layer] = 0 * global_model_before_forget[layer]

        for ii in range(len(new_client_models)):
            old_param_update[layer] += old_client_models[ii][layer]
            new_param_update[layer] += new_client_models[ii][layer]
        old_param_update[layer] /= (ii + 1)  # Model Params： oldCM
        new_param_update[layer] /= (ii + 1)  # Model Params： newCM

        old_param_update[layer] = old_param_update[layer] - global_model_before_forget[
            layer]  # 参数： oldCM - oldGM_t
        new_param_update[layer] = new_param_update[layer] - global_model_after_forget[
            layer]  # 参数： newCM - newGM_t

        step_length = tf.norm(old_param_update[layer])  # ||oldCM - oldGM_t||
        step_direction = new_param_update[layer] / tf.norm(
            new_param_update[layer])  # (newCM - newGM_t)/||newCM - newGM_t||

        return_model_state[layer] = new_global_model_state[layer] + step_length * step_direction

    return_global_model = copy.deepcopy(global_model_after_forget)

    return_global_model.load_state_dict(return_model_state)

    return return_global_model

    # return forget_global_model


def learning():
    LR_A = 0.0001  # learning rate for actor
    LR_C = 0.001  # learning rate for critic
    numberOfRobots = 20
    numberOfAPs = 4
    env = RobotEnv(numOfRobo=numberOfRobots, numOfAPs=numberOfAPs)  # number of robots and AP

    i = 0
    robotics = []  # 机器人
    robot_movers = list()  # 移动的机器人
    APs = []  # 接入点
    for i in range(numberOfRobots):
        Robot0 = Agent(f'actor{i}', f'critic{i}', f'actor{i}_target', f'critic{i}_target', alpha=LR_A, beta=LR_C,
                       input_dims=[148],
                       tau=0.001, n_actions=8)
        robotics.append(Robot0)
        Robot0_mover = Agent(f'actor{i}_mover', f'critic{i}_mover', f'actor{i}_mover_target', f'critic{i}_mover_target',
                             alpha=LR_A,beta=LR_C, input_dims=[148], tau=0.001, n_actions=2)
        robot_movers.append(Robot0_mover)
    for i in range(numberOfAPs):
        AP1 = Agent(f'actor{i}_AP', f'critic{i}_AP', f'actor{i}_AP_target', f'critic{i}_AP_target', alpha=LR_A,
                    beta=LR_C,input_dims=[357], tau=0.001, n_actions=8)
        APs.append(AP1)

    Server_robot = Federated_Server(numberOfRobots,name_actor='server_actor', name_critic='server_critic', input_dims=[148],
                                    n_actions=8,layer1_size=32, layer2_size=32)
    Server_AP = Federated_Server_AP(name_actor='server_actor_AP', name_critic='server_critic_AP', input_dims=[357],
                                    n_actions=8, layer1_size=32, layer2_size=32)
    score_Robot = 0
    score_AP = 0
    score1_history = []
    score0_history = []
    delay_history = []
    energy_history = []
    reject_history = []
    steps = 100
    Explore = 100000.
    warmed = 0
    new_state = 0
    epsilon = 1
    epsilon_move = 1
    all_client_params=list()  #客户端的
    all_global_params=list()  #全局的
    log_dir = "logs/"
    summary_writer = tf.summary.create_file_writer(log_dir)
    cnt=0
    for i in range(10):
        obs = env.reset()
        obs_move = obs
        a = 0
        test = 0  # this is for debugging, and when it is '1', we can choose action to debug
        for move in range(10):
            Local_Observaion = []
            AP_observation = obs[numberOfRobots]  # the last observation is for access points
            for R in range(numberOfRobots):
                Local_Observaion.append(obs[R])
            epsilon_move -= 1 / Explore
            act_robot_moves = np.zeros((numberOfRobots, 2))
            #移动是二维的
            if np.random.random() < epsilon_move:  #贪婪探索
                for x in range(numberOfRobots):
                    act0_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act_robot_moves[x] = act0_robot_move
            else:
                for x in range(numberOfRobots):
                    act0_robot_move = robot_movers[x].choose_action(Local_Observaion[x])
                    act_robot_moves[x] = act0_robot_move

            # start federation loop
            for k in range(1):
                for j in range(10):
                    # while not done:
                    Local_Observaion = []
                    AP_observation = obs[numberOfRobots]  # the last observation is for access points
                    for R in range(numberOfRobots):
                        Local_Observaion.append(obs[R])
                    epsilon -= 1 / Explore
                    a += 1
                    act_robots = np.zeros((numberOfRobots,8))
                    act_APs= np.zeros((numberOfAPs,8))
                    if np.random.random() <= epsilon:
                        for y in range(numberOfRobots):
                            act_robots[i] = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(
                                -1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0,
                                                                                            1.0), np.random.uniform(
                                -1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)

                        for y in range(numberOfAPs):
                            act_APs[y] = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(
                                -1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0,
                                                                                            1.0), np.random.uniform(
                                -1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)

                    else:
                        for y in range(numberOfRobots):
                            act_robots[i] = robotics[y].choose_action(Local_Observaion[y])
                        for y in range(numberOfAPs):
                            act_APs[y] = APs[y].choose_action(AP_observation)

                    new_state, reward_Robot, reward_AP, done, info, accept, AoI, energy, posX0, posY0, posX1, posY1 = env.step_task_offloading(act_APs, act_robots, act_robot_moves)
                    for re in range(numberOfRobots):
                        robotics[re].remember(Local_Observaion[re], act_robots[re], reward_Robot[re], new_state[re], done)
                    for re in range(numberOfAPs):
                        APs[re].remember(AP_observation, act_APs[re], reward_AP[re], new_state[numberOfRobots], done)

                    for re in range(numberOfRobots):
                        robotics[re].learn()
                    for re in range(numberOfAPs):
                        APs[re].learn()

                    score_Robot += np.average(reward_Robot)
                    score_AP += np.average(reward_AP)
                    cnt=cnt+1
                    tf.summary.scalar('Robots reward', score_Robot, step=cnt)
                    tf.summary.scalar('APs reward', score_AP, step=cnt)
                    print("Robots reward = ", score_Robot)
                    print("APs reward = ", score_AP)
                    if i %10 == 0:
                        with open("03-accept_LR_high_action_corrected.txt", 'a') as reward_APs:
                            reward_APs.write(str(accept) + '\n')
                        with open("03-AoI_LR_high_action_corrected.txt", 'a') as AoI_file:
                            AoI_file.write(str(AoI) + '\n')
                        with open("03-energy_LR_high_action_corrected.txt", 'a') as energy_file:
                            energy_file.write(str(energy) + '\n')
                    obs = new_state
                    obs_move = new_state
                # calculate federation
                # start federation
                clients_params = list() #
                for rn in range(numberOfRobots):
                    client0_params=robotics[rn].get_param()
                    clients_params.append(client0_params)
                    #只有当这个的奖励超过平均的才会上传聚合
                    if reward_Robot[rn] >= np.average(reward_Robot):
                        Server_robot.actors_params[rn]=robotics[rn].get_param()
                        Server_robot.robot_sents[rn] = 1
                all_client_params.append(clients_params)
                Server_AP.actor_paramAP1 = APs[0].get_param()
                Server_AP.actor_paramAP2 = APs[1].get_param()
                Server_AP.actor_paramAP3 = APs[2].get_param()
                Server_AP.actor_paramAP4 = APs[3].get_param()
                #聚合
                actor_robot_new = Server_robot.federation()
                actor_AP_new = Server_AP.federation()
                all_global_params.append(actor_robot_new)

                #设置新的参数
                for f in range(numberOfRobots):
                    robotics[f].actor.params = actor_robot_new
                for f in range(numberOfAPs):
                    APs[f].actor.params = actor_AP_new
                #重新置为0
                for f in range(numberOfRobots):
                    Server_robot.robot_sents[f]=0
                # end of federation

            with open("03-reward_robot_LR_high_action_corrected.txt", 'a') as reward_robots:
                reward_robots.write(str(score_Robot) + '\n')
            with open("03-reward_AP_LR_high_action_corrected.txt", 'a') as reward_APs:
                reward_APs.write(str(score_AP) + '\n')
            with open("03-pos_0_high_action_corrected.txt", 'a') as pos0:
                pos0.write(str(posX0) + ', ' + str(posY0) + '\n')
            with open("03-pos_1_high_action_corrected.txt", 'a') as pos1:
                pos1.write(str(posX1) + ', ' + str(posY1) + '\n')

            #移动结束后，移动机器人也需要存储经验加学习
            for mr in range(numberOfRobots):
                robot_movers[mr].remember(Local_Observaion[mr], act_robot_moves[mr], score_Robot, new_state[mr], False)
            for mr in range(numberOfRobots):
                robot_movers[mr].learn()

            score_Robot = 0
            score_AP = 0
    return robotics,robot_movers,APs,all_global_params,all_client_params
if __name__ == '__main__':
    robotics,robot_movers,APs,all_global_params,all_client_params=learning()
    FL_params = Arguments()
    std_time = time.time()
    # Set the parameter IF_unlearning =True so that global_train_once skips forgotten users and saves computing time
    FL_params.if_unlearning = True
    # Set the parameter forget_client_IDx to mark the user's IDX that needs to be forgotten
    FL_params.forget_client_idx = 2
    unlearning(robotics,robot_movers,APs,all_global_params,all_client_params,FL_params)
