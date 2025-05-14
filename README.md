# Twin PPO for Maximizing SEE (Security Energy Efficiency)

This guide explains how to run the **Twin PPO** algorithm to maximize Security Energy Efficiency (SEE).                      
This project is a simple ppo implementation for study.

----

****My Twin-PPO implementation is much better than existing D-PPO paper's performance****       
----
**48.44(paper) -> 57.20(my code) in SEE, [10 trials average]**                                     
>Replay buffer in paper's algorithm decrease the performance of ppo.                       
>Change rollout buffer from replay buffer.                
>Even much faster convergence.

**Existing D-PPO paper**: W. Zhang, R. Zhao and Y. Xu, "Aerial Reconfigurable Intelligent Surface-Assisted Secrecy Energy-Efficient Communication Based on Deep Reinforcement Learning," 2024 12th International Conference on Intelligent Computing and Wireless Optical Communications (ICWOC), Chongqing, China, 2024, pp. 60-65, doi: 10.1109/ICWOC62055.2024.10684922.                         
https://ieeexplore.ieee.org/document/10684922


## Prerequisites
Before running the script, to ensure the following dependencies are installed:
- Python 3.10.x
- Requirements libraries
- Recommend running code on linux

You can install the dependencies by run below command:

```bash
$ pip install -r requirements.txt
```
Train twin-agent
```bash
$ cd Twin-PPO
$ python main_train.py
```
Use metrics to see your models performance
```bash
$ python3 load_and_plot.py --path data/storage/training/[your data path] --ep-num [your episode]
```
---
### Change setup
If you want to change reward or num_epochs, open the train_ppo.py then modify parser to make required=True
#### example)
```python
parser.add_argument('--reward', type=str, required=True, default='see',
                    help="which reward would you like to implement ['ssr', 'see']")
parser.add_argument('--ep-num', type=int, required=True, default=300,
                    help="how many episodes do you want to train yout DRL")
```
and then 
```bash
$ python main_train.py --reward ssr --ep-num 20000
```
---
## Reference
1. https://github.com/yjwong1999/Twin-TD3
2. https://hiddenbeginner.github.io/Deep-Reinforcement-Learnings/book/Chapter2/12-implementation-ppo.html#id4
3. https://arxiv.org/abs/1707.06347
