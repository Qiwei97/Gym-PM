# Gym-PM 

This is a Gym RL Environment for Predictive Maintenance.

Compatible with gym-api, Stable Baselines3, RLlib.

### Installation
```
git clone https://github.com/Qiwei97/Gym-PM.git
cd Gym-PM
pip install -e .
```

### Set up
```
import gym
import gym_pm

env = gym.make('Rail-v2')
```

## Environments

#### Rolling Stock
  * Rail-v1 (Weibull)
  * Rail-v2 (Synthetic Data)

#### Assembly Line
  * Assembly-v1 (Weibull)
  * Assembly-v2 (Synthetic Data)
