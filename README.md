# Twin PPO for Maximizing SEE (Security Energy Efficiency)

This guide explains how to run the **Twin PPO** algorithm to maximize Security Energy Efficiency (SEE).

This project is a simple ppo implementation for study.

~~***"Current version doesn't work now. I trying to find what is wrong."(02/14/25)***~~

***"The issues has been resolved"(02/22/25)***

***"SAC implementation is ongoing"***

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
$ python3 train_ppo_base.py
```
Use metrics to see your models performance(path will be inside of ./data/storage/training/)
```bash
$ python3 load_and_plot.py --path [your data path] --ep-num [your episode]
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
$ python train_ppo_base.py --reward ssr --ep-num 500
```
---
## Reference
1. https://github.com/yjwong1999/Twin-TD3
2. https://hiddenbeginner.github.io/Deep-Reinforcement-Learnings/book/Chapter2/12-implementation-ppo.html#id4
3. https://arxiv.org/abs/1707.06347
4. https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
