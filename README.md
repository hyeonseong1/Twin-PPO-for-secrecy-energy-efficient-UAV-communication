# Twin PPO for Maximizing SEE (Security Energy Efficiency)

This guide explains how to run the **Twin PPO** algorithm to maximize Security Energy Efficiency (SEE).

This project is a simple ppo implementation for study.

## Prerequisites
Before running the script, ensure the following dependencies are installed:
- Python 3.10.x
- Requirements libraries
- Recommend running code on linux

You can install the dependencies by running:

```bash
$ pip install -r requirements.txt
```
Then you can easily train agents
```bash
$ cd Twin-PPO
$ python train_ppo.py
```
---
### Change setup
If you want to change reward or num_epochs, open the train_ppo.py then modify parser to make required=True
#### example)
```python
parser.add_argument('--reward', type=str, required=True, default='see',
                    help="which reward would you like to implement ['ssr', 'see']")
parser.add_argument('--ep-num', type=int, required=True, default=1000,
                    help="how many episodes do you want to train yout DRL")
```
and then 
```bash
$ python train_ppo.py --reward ssr --ep-num 1000
```
---
## Reference
1. https://github.com/yjwong1999/Twin-TD3
2. https://hiddenbeginner.github.io/Deep-Reinforcement-Learnings/book/Chapter2/12-implementation-ppo.html#id4
