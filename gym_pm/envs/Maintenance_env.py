import gym
from gym import spaces
import time
import numpy as np
import pandas as pd
from gym_pm.envs.Objects import Machine
from IPython.display import display, clear_output

class PM_Env(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self, env_config=None):

        # Initialize Everything
        self.time_step = 0

        # Prepare Game Objects
        self.machine_a = Machine(output_rate=1, alpha=10)
        self.machine_b = Machine(output_rate=1, alpha=15)
        self.machines = [self.machine_a, self.machine_b]

        self.max_duration = 60
        self.max_resource = 100

        # action space
        self.action_space = spaces.Discrete(2 * len(self.machines))
        # obs space
        self.observation_space = spaces.Dict({
                "age": spaces.Box(low=0., high=self.max_duration, shape=(len(self.machines),), dtype=np.float32),
                "condition": spaces.MultiBinary(len(self.machines)),
                "resources": spaces.Box(low=0., high=self.max_resource, shape=(len(self.machines),), dtype=np.float32),
                "survival_prob": spaces.Box(low=0., high=1., shape=(len(self.machines),), dtype=np.float32)
                })

    def reset(self):

        self.time_step = 0
        self.machine_a = Machine(output_rate=1, alpha=10)
        self.machine_b = Machine(output_rate=1, alpha=15)
        self.machines = [self.machine_a, self.machine_b]

        return self.observation()

    def observation(self):

        state = {
            "age": [],
            "condition": [],
            "resources": [],
            "survival_prob": []
        }

        for machine in self.machines:
            state['age'].append(machine.age)
            state['condition'].append(machine.working)
            state['resources'].append(machine.capacity)
            state['survival_prob'].append(machine.survival_prob)

        state = {i: np.array(j, dtype='float32') for (i, j) in state.items()}

        return state

    def get_reward(self):
            
        # reward = 0
        # for machine in self.machines:
        #     reward += machine.output * machine.product_price # Revenue
        #     reward -= machine.downtime * machine.output_rate * machine.product_price # lost sales
        #     reward -= machine.repair_cost * machine.repair_counter # repair

        reward = 100
        for machine in self.machines:
            reward -= machine.repair_cost * machine.repair_status * machine.repair_time
            if machine.working == False:
                reward -= 120

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
            # Inventory 
            machine.update_inv()
            # Reset Repair Status
            machine.repair_status = 0

        # Interactions
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
            result.resources = result.resources.astype(float).round(2)
            result['ttf'] = [machine.ttf[0] for machine in self.machines]
            result['repair_count'] = [machine.repair_counter for machine in self.machines]
            result['reward'] = self.get_reward()
            result['time'] = int(self.time_step)
            
            clear_output(wait=True)
            display(result)
            time.sleep(1)

    def close(self):
        pass