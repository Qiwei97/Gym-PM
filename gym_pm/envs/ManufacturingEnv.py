import gym
from gym import spaces
import time
import numpy as np
import pandas as pd
from gym_pm.envs.Objects import Factory, Factory_v2
from IPython.display import display, clear_output

class Assembly_Env(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self, env_config=None):

        # Episode length
        self.episode_length = 100 # max timestep
        self.max_resource = 10000 # Set to a high value

        # Initialize everything
        self.reset()

        # action space
        self.action_space = spaces.Discrete(4 * len(self.machines) - 1)
        # obs space
        self.observation_space = spaces.Dict({
                "age": spaces.Box(low=0., high=self.episode_length, shape=(len(self.machines),), dtype=np.float32),
                "condition": spaces.MultiBinary(len(self.machines)),
                "resources": spaces.Box(low=0., high=self.max_resource, shape=(len(self.machines),), dtype=np.float32),
                "survival_prob": spaces.Box(low=0., high=1., shape=(len(self.machines),), dtype=np.float32),
                "backlog": spaces.Box(low=0., high=self.max_resource, shape=(1,), dtype=np.float32)
                })

    def reset(self):

        # reset time_step
        self.time_step = 0
        # reset backlog
        self.backlog = 0

        # Prepare Objects (Add more objects as desired)
        # We use 1 in this example
        self.machine_a = Factory(output_rate=3, alpha=10, episode_length=self.episode_length)
        self.machines = [self.machine_a]

        return self.observation()

    def observation(self):

        state = {
            "age": [],
            "condition": [],
            "resources": [],
            "survival_prob": [],
            "backlog": [self.backlog]
        }

        for machine in self.machines:
            state['age'].append(machine.age)
            state['condition'].append(machine.working)
            state['resources'].append(machine.capacity)
            state['survival_prob'].append(machine.survival_prob)

        state = {i: np.array(j, dtype='float32') for (i, j) in state.items()}

        return state

    def get_reward(self):

        reward = 0.
        for machine in self.machines:
            # Repair Cost
            reward -= machine.repair_cost * machine.repair_status * machine.repair_time
            # Resupply Cost
            reward -= machine.resupply_cost * machine.resupply_status * machine.resupply_qty
            # Inventory Cost
            reward -= machine.capacity * machine.storage_cost
            reward -= machine.output * machine.storage_cost
            # Sales Revenue
            reward += machine.fulfilled_orders * machine.product_price
            if machine.working == False:
                reward -= 100

        return reward

    def check_done(self):

        if self.time_step >= self.episode_length:
            done = True
        else:
            done = False

        return done

    def get_demand(self, machine):

        self.backlog += machine.demand_dist[self.time_step - 1]

        if self.backlog > machine.output:
            self.backlog -= machine.output
            machine.fulfilled_orders = machine.output
            machine.output = 0
        else:
            machine.output -= self.backlog 
            machine.fulfilled_orders = self.backlog
            self.backlog = 0

    def step(self, action):

        self.time_step += 1

        for machine in self.machines:
            # Replenish Stock
            machine.update_LT()
            # Deterioriation
            machine.failure_check()
            # Inventory 
            machine.update_inv()
            # Fulfil Demand
            self.get_demand(machine)
            # Reset Status
            machine.repair_status = 0
            machine.repair_status = 0

        # Interactions (Add more as desired)
        if action == 0:
            self.machine_a.repair()
        elif action == 1:
            self.machine_a.resupply()
        else:
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
            result.resources = result.resources.astype(float).round(2)
            result['ttf'] = [machine.ttf[0] for machine in self.machines]
            result['repair_count'] = [machine.repair_counter for machine in self.machines]
            result['reward'] = self.get_reward()
            # result['time'] = int(self.time_step)
            result['lead_time'] = [machine.resupply_list for machine in self.machines]
            result['backlog'] = result['backlog'].astype(int)

            clear_output(wait=True)
            display(result)
            time.sleep(1)

    def close(self):
        pass


class Assemblyv2_Env(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self, env_config=None, output_rate=2, 
                 data='PdM1', capacity=20,
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

        self.backlog += self.machine.df.iloc[self.time_step]['demand']

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

    def render(self, mode="console"):

        if mode == "console":
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
            
            clear_output(wait=True)
            display(result)
            time.sleep(1)
            
    def close(self):
        pass
