import gym
from gym import spaces
import time
import numpy as np
import pandas as pd
from gym_pm.envs.Objects import Factory, Factory_v2
from IPython.display import display, clear_output


class Assembly_Env(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self, env_config=None, output_rate=2, 
                 alpha=10, beta=5, capacity=20,
                 repair_cost=30, resupply_cost=3, 
                 storage_cost=2, resupply_qty=10, 
                 lead_time=3, product_price=75):

        # Cost elements
        self.output_rate = output_rate
        self.capacity = capacity
        self.repair_cost = repair_cost
        self.resupply_cost = resupply_cost
        self.storage_cost = storage_cost
        self.resupply_qty = resupply_qty
        self.lead_time = lead_time
        self.product_price = product_price

        # Episode length
        self.max_duration = 1000 # max time

        # Initialize everything
        self.alpha = alpha
        self.beta = beta
        self.reset()

        self.max_resource = self.machine.output_rate * 1000 # Set to a high value

        # action space
        self.action_space = spaces.Discrete(3)
        # obs space
        self.observation_space = spaces.Dict({
                "age": spaces.Box(low=0., high=self.max_duration, shape=(1,), dtype=np.float32),
                "Failure": spaces.MultiBinary(1),
                "resources": spaces.Box(low=0., high=self.max_resource, shape=(1,), dtype=np.float32),
                "backlog": spaces.Box(low=0., high=self.max_resource, shape=(1,), dtype=np.float32)
                })

    def reset(self):

        # reset timer
        self.timer = 0      
        # reset backlog
        self.backlog = 0

        self.machine = Factory(output_rate=self.output_rate, 
                               alpha=self.alpha,
                               beta=self.beta,
                               capacity=self.capacity,
                               episode_length=self.max_duration,                 
                               repair_cost=self.repair_cost, 
                               resupply_cost=self.resupply_cost,
                               storage_cost=self.storage_cost, 
                               resupply_qty=self.resupply_qty,
                               lead_time=self.lead_time, 
                               product_price=self.product_price)

        return self.observation()

    def observation(self):

        state = {
            "age": [self.machine.age],
            "Failure": [-(self.machine.working - 1)],
            "resources": [self.machine.capacity],
            "backlog": [self.backlog]
        }

        state = {i: np.array(j, dtype='float32') for (i, j) in state.items()}

        return state

    def get_reward(self):

        reward = 0.
        # Repair Cost
        reward -= self.machine.repair_cost * self.machine.repair_status * self.machine.repair_time
        # Resupply Cost
        reward -= self.machine.resupply_cost * self.machine.resupply_status * self.machine.resupply_qty
        # Inventory Cost
        reward -= self.machine.capacity * self.machine.storage_cost
        reward -= self.machine.output * self.machine.storage_cost
        # Sales Revenue
        reward += self.machine.fulfilled_orders * self.machine.product_price
        if self.machine.working == False:
            reward -= 200

        return reward

    def check_done(self):

        if self.timer >= self.max_duration:
            done = True
        else:
            done = False

        return done

    def fulfil_demand(self):

        self.backlog += self.machine.demand_dist[self.timer]

        if self.machine.output == 0:
            return
        elif self.backlog > self.machine.output:
            self.backlog -= self.machine.output
            self.machine.fulfilled_orders = self.machine.output
            self.machine.output = 0
        else:
            self.machine.output -= self.backlog 
            self.machine.fulfilled_orders = self.backlog
            self.backlog = 0

    def step(self, action):

        # Reset Status
        self.machine.repair_status = 0
        self.machine.resupply_status = 0

        # Replenish Stock
        self.machine.update_lt()
        # Deterioriation
        self.machine.failure_check()
        # Inventory
        self.machine.update_inv()
        # Reduce Backlog
        self.fulfil_demand()

        # Interactions (Add more as desired)
        if action == 0:
            self.machine.repair()
        elif action == 1:
            self.machine.resupply()
        else:
            pass

        obs = self.observation()
        reward = self.get_reward()
        done = self.check_done()
        info = {}

        self.timer += 1

        return obs, reward, done, info

    def render(self, mode="console"):

        result = pd.Series({i: j[0] for (i, j) in self.observation().items()})

        result['age'] = result['age'].astype(int)
        result.Failure = result.Failure.astype(bool)
        result['ttf'] = self.machine.ttf[0]
        result['repair_count'] = self.machine.repair_counter
        result['reward'] = self.get_reward()
        result['duration'] = int(self.timer)
        result['lead_time'] = self.machine.resupply_list
        result['backlog'] = self.backlog
        result = result.to_frame('Results')
            
        if mode == 'human':

            clear_output(wait=True)
            display(result)
            time.sleep(1)

        return result

    def close(self):
        pass


class Assemblyv2_Env(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self, env_config=None, output_rate=2, 
                 data='PdM2', split='Train', capacity=20,
                 repair_cost=30, resupply_cost=3, 
                 storage_cost=2, resupply_qty=10, 
                 lead_time=3, product_price=75):

        # Cost elements
        self.output_rate = output_rate
        self.capacity = capacity
        self.repair_cost = repair_cost
        self.resupply_cost = resupply_cost
        self.storage_cost = storage_cost
        self.resupply_qty = resupply_qty
        self.lead_time = lead_time
        self.product_price = product_price

        # Initialize everything
        self.data = data
        self.split = split
        self.reset()

        # Episode length
        self.max_duration = len(self.machine.df) # max time
        self.max_resource = self.machine.output_rate * 1000 # Set to a high value

        # action space
        self.action_space = spaces.Discrete(3)

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

        obs_space['backlog'] = spaces.Box(low=0., high=self.max_resource, shape=(1,), dtype=np.float32)
        obs_space['resources'] = spaces.Box(low=0., high=self.max_resource, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Dict(obs_space)

    def reset(self):

        # reset timer
        self.timer = 0
        # reset time_step
        self.time_step = 0        
        # reset backlog
        self.backlog = 0

        self.machine = Factory_v2(output_rate=self.output_rate, 
                                  data=self.data,
                                  split=self.split,
                                  capacity=self.capacity,                 
                                  repair_cost=self.repair_cost, 
                                  resupply_cost=self.resupply_cost,
                                  storage_cost=self.storage_cost, 
                                  resupply_qty=self.resupply_qty,
                                  lead_time=self.lead_time, 
                                  product_price=self.product_price)

        return self.observation()

    def observation(self):

        state = self.machine.df.iloc[self.time_step].to_dict()
        state['backlog'] = self.backlog
        state['resources'] = self.machine.capacity
        state.pop('ttf')
        state = {i: np.array([j], dtype='float32') for (i, j) in state.items()}

        return state

    def get_reward(self):

        reward = 0.
        # Repair Cost
        reward -= self.machine.repair_cost * self.machine.repair_status * self.machine.repair_time
        # Resupply Cost
        reward -= self.machine.resupply_cost * self.machine.resupply_status * self.machine.resupply_qty
        # Inventory Cost
        reward -= self.machine.capacity * self.machine.storage_cost
        reward -= self.machine.output * self.machine.storage_cost
        # Sales Revenue
        reward += self.machine.fulfilled_orders * self.machine.product_price
        if self.machine.working == False:
            reward -= 200

        return reward

    def check_done(self):

        if self.timer >= self.max_duration:
            done = True
        else:
            done = False

        return done

    def fulfil_demand(self):

        self.backlog += self.machine.demand_dist[self.timer]

        if self.machine.output == 0:
            return
        elif self.backlog > self.machine.output:
            self.backlog -= self.machine.output
            self.machine.fulfilled_orders = self.machine.output
            self.machine.output = 0
        else:
            self.machine.output -= self.backlog 
            self.machine.fulfilled_orders = self.backlog
            self.backlog = 0

    def step(self, action):
       
        # Reset Status
        self.machine.repair_status = 0
        self.machine.resupply_status = 0

        # Replenish Stock
        self.machine.update_lt()
        # Deterioriation
        self.machine.failure_check(self.time_step)
        # Inventory
        self.machine.update_inv()
        # Reduce Backlog
        self.fulfil_demand()

        # Interactions (Add more as desired)
        if action == 0:
            self.machine.repair()
            # Reset time_step
            self.time_step = np.random.choice(self.machine.df[self.machine.df.age == 1].index)
        elif action == 1:
            self.machine.resupply()
        else:
            pass

        obs = self.observation()
        reward = self.get_reward()
        done = self.check_done()
        info = {}

        self.timer += 1

        if self.machine.working:
            self.time_step += 1

        return obs, reward, done, info

    def render(self, mode='console'):

        result = self.machine.df.iloc[self.time_step][['age', 'ttf', 'Failure']]

        result.Failure = result.Failure.astype(bool)
        result['resources'] = round(self.machine.capacity, 2)
        result['repair_count'] = self.machine.repair_counter
        result['reward'] = self.get_reward()
        result['time_step'] = int(self.time_step)
        result['duration'] = int(self.timer)
        result['lead_time'] = self.machine.resupply_list
        result['backlog'] = self.backlog
        result = result.to_frame('Results')
            
        if mode == 'human':

            clear_output(wait=True)
            display(result)
            time.sleep(1)

        return result
            
    def close(self):
        pass
