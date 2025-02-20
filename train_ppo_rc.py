import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 라이브러리 중복 오류 검사

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--reward', type=str, required=False, default='see',
                    help="which reward would you like to implement ['ssr', 'see']")
parser.add_argument('--ep-num', type=int, required=False, default=1000,
                    help="how many episodes do you want to train yout DRL")
parser.add_argument('--model', type=str, required=False, default='ppo_rc',
                    help="two models are available ['ppo_rc', 'ppo_per']")
parser.add_argument('--trained-uav', default=False, action='store_true',
                    help='use trained uav instead of retraining')

args = parser.parse_args()
REWARD_DESIGN = args.reward
EPISODE_NUM = args.ep_num
TRAINED_UAV = args.trained_uav

assert REWARD_DESIGN in ['ssr', 'see'], "reward must be ['ssr', 'see']"

import numpy as np
import math
import time
import torch
from environment.env import MiniSystem

if args.model == 'ppo_per':
    from ppo_per import PPOAgent
else:
    from ppo_rc import PPOAgent

episode_num = EPISODE_NUM
episode_cnt = 0
step_num = 100
project_name = f'pre-trained_uav/PPO_{REWARD_DESIGN}' if TRAINED_UAV else f'training/PPO_{REWARD_DESIGN}'

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

# 기본 시드 고정
seed = 42
np.random.seed(seed)
np.random.seed(seed)
# 파이토치
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

agent_ris = PPOAgent(
    alpha = 1e-4,
    beta = 1e-3,
    input_dim = system.get_system_state_dim(),
    n_action = system.get_system_action_dim() - 2,
    lamda = 1,
    gamma = 0.99,
    epsilon = 1e-6,
    max_size = 1000000,
    layer1_size = 400,
    layer2_size = 256,
    layer3_size = 128,
    batch_size = 64,
    update_actor_interval = 2,
    noise = 'AWGN'
)
agent_uav = PPOAgent(
    alpha = 1e-4,
    beta = 1e-3,
    input_dim = 3,
    n_action = 2,
    lamda = 1,
    gamma = 0.99,
    epsilon = 1e-6,
    max_size = 1000000,
    layer1_size = 400,
    layer2_size = 256,
    layer3_size = 128,
    batch_size = 64,
    update_actor_interval = 2,
    noise = 'AWGN'
)

if TRAINED_UAV:
    benchmark = f'data/storage/benchmark/PPO_{REWARD_DESIGN}_benchmark'
    agent_uav.load_models(
        load_file_actor=benchmark + '/Actor_UAV_ppo',
        load_file_critic=benchmark + '/Critic_UAV_ppo'
    )


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
    else:
        observarion_ris = system.observe()
    observarion_uav = list(system.UAV.coordinate)
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
            # action_ris = agent_ris.act(observarion_ris, greedy=0.5)
            # action_uav = agent_uav.act(observarion_uav, greedy=0.5)
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
            # 5 update PPO by old policy
            agent_ris.learn()
            if not TRAINED_UAV:
                agent_uav.learn()

            # system.render_obj.render(0.001) # no rendering for faster
            observarion_ris = next_state_ris
            observarion_uav = next_state_uav
            if done == True:
                break

        else:
            # system.render_obj.render_pause()  # no rendering for faster
            time.sleep(0.001)  # time.sleep(1)
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


import matplotlib.pyplot as plt

x = list(range(episode_num))
plt.plot(x, see)
plt.title('SEE (not averaged)')
plt.xlabel('epoch')
plt.ylabel('SEE')
plt.savefig('/result_png/SEE_ppo_rc.png')
