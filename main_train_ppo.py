import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 라이브러리 중복 오류 검사

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--reward', type=str, required=False, default='see',
                    help="which reward would you like to implement ['ssr', 'see']")
parser.add_argument('--ep-num', type=int, required=False, default=300,
                    help="how many episodes do you want to train yout DRL")
parser.add_argument('--trained-uav', default=False, action='store_true',
                    help='use trained uav instead of retraining')

args = parser.parse_args()
REWARD_DESIGN = args.reward
TRAINED_UAV = args.trained_uav

assert REWARD_DESIGN in ['ssr', 'see'], "reward must be ['ssr', 'see']"

import numpy as np
import math
import time
from environment.env import MiniSystem
from ppo import PPOAgent

episode_num = args.ep_num
episode_cnt = 0
step_num = 100 # 128
batch_size = 100

lr1 = 3e-4  # 1e-4
lr2 = 3e-3  # 1e-3
lr_step_size = 300
decay_rate = 0.5
k_epoch = 10
eps_clip = 0.2

project_name = f'pre-trained_uav/PPO_{REWARD_DESIGN}' if TRAINED_UAV else f'PPO/ppo_{REWARD_DESIGN}_{args.ep_num}_{k_epoch}'

system = MiniSystem(
    user_num=2,
    RIS_ant_num=4,
    UAV_ant_num=4,
    if_dir_link=1,
    if_with_RIS=True,
    if_move_users=True,
    if_movements=True,
    reverse_x_y=(False, False),
    if_UAV_pos_state=True,
    reward_design=REWARD_DESIGN,
    project_name=project_name,
    step_num=step_num
)

if_Theta_fixed = False
if_G_fixed = False
if_BS = False
if_robust = True


agent_ris = PPOAgent(
    alpha = lr1,
    beta = lr2,
    input_dim = system.get_system_state_dim(),
    n_action = system.get_system_action_dim() - 2,
    lamda = 0.95,
    gamma = 0.99,
    eps_clip = eps_clip,
    layer1_size = 256,
    layer2_size = 128,
    batch_size = batch_size,
    K_epochs = k_epoch,
    noise = 'AWGN'
)
agent_uav = PPOAgent(
    alpha = lr1,
    beta = lr2,
    input_dim = 3,
    n_action = 2,
    lamda = 0.95,
    gamma = 0.99,
    eps_clip = eps_clip,
    layer1_size = 256,
    layer2_size = 128,
    batch_size = batch_size,
    K_epochs = k_epoch,
    noise = 'AWGN'
)

if TRAINED_UAV:
    benchmark = f'data/storage/benchmark/ppo_{REWARD_DESIGN}_benchmark'
    agent_uav.load_models(
        load_file_actor=benchmark + '/Actor_UAV_ppo',
        load_file_critic=benchmark + '/Critic_UAV_ppo'
    )


from datetime import datetime

start_time = datetime.now()

print("***********************traning information******************************")
see = []
while episode_cnt < episode_num:
    # 1 reset the whole system
    system.reset()
    step_cnt = 0
    score_per_ep = 0

    # 2 get the initial state
    if if_robust:
        tmp = system.observe()
        # z = np.random.multivariate_normal(np.zeros(2), 0.5*np.eye(2), size=len(tmp)).view(np.complex128)
        z = np.random.normal(size=len(tmp))
        observarion_ris = list(
            np.array(tmp) + 0.6 * 1e-7 * z
        )
        # print(observarion_ris)
        # observarion_ris = np.clip(observarion_ris, -1e3, 1e3)
    else:
        observarion_ris = system.observe()
    observarion_uav = list(system.UAV.coordinate)
    # observarion_uav = np.clip(observarion_uav, -1e3, 1e3)

    while step_cnt < step_num:
        # 1 count num of step in one episode
        step_cnt += 1
        # judge if pause the whole system
        if not system.render_obj.pause:
            # 2 choose action acoording to current state
            action_ris = agent_ris.act(observarion_ris, greedy=0.1 * math.pow(
                (1 - episode_cnt / episode_num), 2))  # 0.1 is action noise factor
            action_uav = agent_uav.act(observarion_uav, greedy=0.1 * math.pow(
                (1 - episode_cnt / episode_num), 2))
            if if_BS:
                action_uav[0] = 0
                action_uav[1] = 0

            if if_Theta_fixed:
                action_ris[0 + 2 * system.UAV.ant_num * system.user_num:] = len(
                    action_ris[0 + 2 * system.UAV.ant_num * system.user_num:]) * [0]

            if if_G_fixed:
                action_ris[0:0 + 2 * system.UAV.ant_num * system.user_num] = np.array(
                    [-0.0313, -0.9838, 0.3210, 1.0, -0.9786, -0.1448, 0.3518, 0.5813, -1.0, -0.2803, -0.4616, -0.6352,
                     -0.1449, 0.7040, 0.4090, -0.8521]) * math.pow(episode_cnt / episode_num, 2) * 0.7
                # action_ris[0:0+2 * system.UAV.ant_num * system.user_num]=len(action_ris[0:0+2 * system.UAV.ant_num * system.user_num])*[0.5]
            # 3 get newstate, reward
            if system.if_with_RIS:
                next_state_ris, reward, done, info = system.step(
                    action_0=action_uav[0],
                    action_1=action_uav[1],
                    G=action_ris[0:0 + 2 * system.UAV.ant_num * system.user_num],
                    Phi=action_ris[0 + 2 * system.UAV.ant_num * system.user_num:],
                    set_pos_x=action_uav[0],
                    set_pos_y=action_uav[1]
                )
                next_state_uav = list(system.UAV.coordinate)
            else:
                next_state_ris, reward, done, info = system.step(
                    action_0=action_uav[0],
                    action_1=action_uav[1],
                    G=action_ris[0:0 + 2 * system.UAV.ant_num * system.user_num],
                    set_pos_x=action_uav[0],
                    set_pos_y=action_uav[1]
                )
                next_state_uav = list(system.UAV.coordinate)

            score_per_ep += reward
            # 4 store state pair into memory pool
            agent_ris.store_transition(observarion_ris, action_ris, reward, next_state_ris, int(done))
            agent_uav.store_transition(observarion_uav, action_uav, reward, next_state_uav, int(done))

            observarion_ris = next_state_ris
            observarion_uav = next_state_uav
            if done == True:
                break
            else:
                # system.render_obj.render_pause()  # no rendering for faster
                time.sleep(0.001)  # time.sleep(1)
    # 5 update agent when buffer is full
    agent_ris.learn()
    if not TRAINED_UAV:
        agent_uav.learn()
    system.data_manager.save_file(episode_cnt=episode_cnt)
    system.reset()
    print("ep_num: " + str(episode_cnt) + "   ep_score:  " + str(score_per_ep))
    see.append(score_per_ep)
    episode_cnt += 1
    # if episode_cnt % 10 == 0:
    #     agent_ris.save_models()
    #     agent_uav.save_models()


# # save the last model
# agent_ris.save_models()
# agent_uav.save_models()
print("***********************time******************************")
end_time = datetime.now()
elapsed_time = end_time - start_time
print(f'training time elapsed: {elapsed_time}')

import matplotlib.pyplot as plt

xaxis = list(range(episode_num))
plt.plot(xaxis, see)
plt.title('reward (not averaged)')
plt.xlabel('epoch')
plt.ylabel('SEE')
plt.savefig(f'./result_png/SEE_ppo_base.png')
plt.show()
