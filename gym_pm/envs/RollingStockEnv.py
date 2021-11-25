import gym
from gym import spaces
import time
import numpy as np
import pandas as pd
from gym_pm.envs.Objects import Train, Train_v2
from IPython.display import display, clear_output

class Rail_Env(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self, env_config=None):

        # Initialize everything
        self.reset()

        # Episode length
        self.max_duration = 60 # max timestep

        # action space
        self.action_space = spaces.Discrete(2 * len(self.machines))
        # obs space
        self.observation_space = spaces.Dict({
                "age": spaces.Box(low=0., high=self.max_duration, shape=(len(self.machines),), dtype=np.float32),
                "condition": spaces.MultiBinary(len(self.machines))
                # "survival_prob": spaces.Box(low=0., high=1., shape=(len(self.machines),), dtype=np.float32)
                })

    def reset(self):

        # reset time_step
        self.time_step = 0

        # Prepare Objects (Add more objects as desired)
        # We use 2 in this example
        self.machine_a = Train(alpha=10)
        self.machine_b = Train(alpha=15)
        self.machines = [self.machine_a, self.machine_b]

        return self.observation()

    def observation(self):

        state = {
            "age": [],
            "condition": []
            # "survival_prob": []
        }

        for machine in self.machines:
            state['age'].append(machine.age)
            state['condition'].append(machine.working)
            # state['survival_prob'].append(machine.survival_prob)

        state = {i: np.array(j, dtype='float32') for (i, j) in state.items()}

        return state

    def get_reward(self):

        reward = 100
        for machine in self.machines:
            # Repair Cost
            reward -= machine.repair_cost * machine.repair_status * machine.repair_time
            if machine.working == False:
                reward -= 200

        return reward

    def check_done(self):

        if self.time_step >= self.max_duration:
            done = True
        else:
            done = False

        return done

    def step(self, action):

        self.time_step += 1

        for machine in self.machines:
            # Deterioriation
            machine.failure_check()
            # Reset Repair Status
            machine.repair_status = 0

        # Interactions (Add more as desired)
        if action == 0:
            self.machine_a.repair()
        if action == 1:
            self.machine_b.repair()
        if action == 2:
            self.machine_a.repair()
            self.machine_b.repair()
        if action == 3:
            pass

        obs = self.observation()
        reward = self.get_reward()
        done = self.check_done()
        info = {}

        return obs, reward, done, info

    def render(self, mode="console"):

        if mode == "console":
            result = pd.DataFrame(self.observation())

            result['age'] = result['age'].astype(int)
            result.condition = result.condition.astype(bool)
            result['ttf'] = [machine.ttf[0] for machine in self.machines]
            result['repair_count'] = [machine.repair_counter for machine in self.machines]
            result['reward'] = self.get_reward()
            result['time'] = int(self.time_step)
            
            clear_output(wait=True)
            display(result)
            time.sleep(1)

    def close(self):
        pass


class Railv2_Env(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self, env_config=None, data='PdM2'):

        # Initialize everything
        self.data = data
        self.reset()

        # Episode length
        self.max_duration = len(self.machine.df) # max time

        # action space
        self.action_space = spaces.Discrete(2)

        # obs space
        obs_bound = pd.DataFrame()
        obs_bound['high'] = self.machine.df.max()
        obs_bound['low'] = self.machine.df.min()
        obs_bound = obs_bound.to_dict(orient='index')
        obs_bound.pop('ttf')

        obs_space = {}
        for i, j in obs_bound.items():
            if i == 'Failure':
                obs_space[i] = spaces.MultiBinary(1)
            else:
                obs_space[i] = spaces.Box(low=j['low'], high=j['high'], shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Dict(obs_space)

    def reset(self):

        # reset timer
        self.timer = 0
        # reset time_step
        self.time_step = 0

        self.machine = Train_v2(self.data)

        return self.observation()

    def observation(self):
        
        state = self.machine.df.iloc[self.time_step].to_dict()
        state.pop('ttf')
        state = {i: np.array([j], dtype='float32') for (i, j) in state.items()}

        return state

    def get_reward(self):

        reward = 0
        if self.machine.repair_status == False:
            reward += 100
        # Repair Cost
        reward -= self.machine.repair_cost * self.machine.repair_status * self.machine.repair_time
        if self.machine.working == False:
            reward -= 200

        return reward

    def check_done(self):

        if self.timer >= self.max_duration:
            done = True
        else:
            done = False

        return done

    def step(self, action):

        # Reset Repair Status
        self.machine.repair_status = 0

        # Deterioriation
        self.machine.failure_check(self.time_step)

        # Interactions (Add more as desired)
        if action == 0:
            self.machine.repair()
            # Reset time_step
            self.time_step = np.random.choice(self.machine.df[self.machine.df.age == 1].index)
        if action == 1:
            pass

        obs = self.observation()
        reward = self.get_reward()
        done = self.check_done()
        info = {}

        self.timer += 1

        if self.machine.working:
            self.time_step += 1

        return obs, reward, done, info

    def render(self, mode="console"):

        if mode == "console":
            result = self.machine.df.iloc[self.time_step][['age', 'ttf', 'Failure']]

            result.Failure = result.Failure.astype(bool)
            result['repair_count'] = self.machine.repair_counter
            result['reward'] = self.get_reward()
            result['time_step'] = int(self.time_step)
            result['duration'] = int(self.timer)
            result = result.to_frame('Results')
            
            clear_output(wait=True)
            display(result)
            time.sleep(1)

    def close(self):
        pass

