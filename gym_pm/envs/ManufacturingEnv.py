import gym
from gym import spaces
import time
import numpy as np
import pandas as pd
from gym_pm.envs.Objects import Factory
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
        if action == 1:
            self.machine_a.resupply()
        if action == 2:
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