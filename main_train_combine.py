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
parser.add_argument('--algo', type=str, required=False, default='sac',
                    help="the algorithm must be ppo or sac")

args = parser.parse_args()
REWARD_DESIGN = args.reward
TRAINED_UAV = args.trained_uav

assert REWARD_DESIGN in ['ssr', 'see'], "reward must be ['ssr', 'see']"

import numpy as np
import math
import time
from environment.env import MiniSystem
from ppo import PPOAgent
# from sac.model_per import *
from sac.model import *
from sac.network import *

episode_num = args.ep_num
episode_cnt = 0
step_num = 100 # 128
batch_size = 256
entropy_rate = 0.3
learning_rate = 3e-4
gradient_step = 2

project_name = f'pre-trained_uav/PPO_{REWARD_DESIGN}' if TRAINED_UAV else f'Combined/{args.algo}_{REWARD_DESIGN}_batch{batch_size}_step{gradient_step}'

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

# RIS control
agent_ris = PPOAgent(
    alpha = learning_rate,
    beta = learning_rate * 10,
    input_dim = system.get_system_state_dim(),
    n_action = system.get_system_action_dim() - 2,
    lamda = 0.95,
    gamma = 0.99,
    eps_clip = 0.2,
    layer1_size = 256,
    layer2_size = 128,
    batch_size = 100,
    K_epochs = 10,
    noise = 'AWGN'
)

# UAV control
agent_uav = SAC(
    state_dim = 3,
    n_action = 2,
    gamma = 0.99,
    tau = 0.005,
    alpha = entropy_rate,
    total_episodes = args.ep_num,
    hidden_dim = 128,
    learning_rate = learning_rate,
    hidden_size = 128, # 128
    max_size = 100000,
    batch_size = batch_size,
)



from datetime import datetime

start_time = datetime.now()

print("***********************traning information******************************")
while episode_cnt < episode_num:
    # 1 reset the whole system
    system.reset()
    step_cnt = 0
    score_per_ep = 0

    # 2 get the initial state of ris and uav
    tmp = system.observe()
    z = np.random.normal(size=len(tmp))
    observarion_ris = list(
        np.array(tmp) + 0.6 * 1e-7 * z
    )
    observarion_uav = list(system.UAV.coordinate)

    # update entropy temperature
    agent_ris.current_episode = episode_cnt
    agent_uav.current_episode = episode_cnt
    # agent_ris.update_alpha()
    agent_uav.update_alpha()

    while step_cnt < step_num:
        # 1 count number of step in one episode
        step_cnt += 1
        # judge if pause the whole system
        if not system.render_obj.pause:
            # 2 choose action according to current state
            action_ris = agent_ris.act(observarion_ris)  # sac don't need any additional noise
            action_uav = agent_uav.act(observarion_uav)
            # 3 get new_state, reward
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
            # 4 store one-step experience
            agent_ris.store_transition(observarion_ris, action_ris, reward, next_state_ris, int(done))
            agent_uav.store_transition(observarion_uav, action_uav, reward, next_state_uav, int(done))
            # system.render_obj.render(0.001) # no rendering for faster
            observation_ris = next_state_ris
            observation_uav = next_state_uav
            if done == True:
                break
            else:
                # system.render_obj.render_pause()  # no rendering for faster
                time.sleep(0.001)  # time.sleep(1)
            # 5 update agent
            if episode_cnt >= 10:
                for _ in range(gradient_step):
                    if not TRAINED_UAV:
                        agent_uav.learn()
    if episode_cnt != 0:
        agent_ris.learn()

    system.data_manager.save_file(episode_cnt=episode_cnt)
    system.reset()
    print("ep_num: " + str(episode_cnt) + "   ep_score:  " + str(score_per_ep))
    episode_cnt += 1


# # save the last model
# agent_ris.save_models()
# agent_uav.save_models()
print("***********************time******************************")
end_time = datetime.now()
elapsed_time = end_time - start_time
print(f'training time elapsed: {elapsed_time}')

