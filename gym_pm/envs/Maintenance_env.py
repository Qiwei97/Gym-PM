import gym
from gym import spaces
import time
import numpy as np
import pandas as pd
from gym_pm.envs.Objects import Machine
from IPython.display import display, clear_output

class PM_Env(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self):

        # Initialize Everything
        self.timer = time.time()

        # Prepare Game Objects
        self.machine_a = Machine(output_rate=1, param=10)
        self.machine_b = Machine(output_rate=1, param=15)
        self.machines = [self.machine_a, self.machine_b]

        self.max_duration = 30
        self.max_resource = 25

        # action space
        self.action_space = spaces.Discrete(2 * len(self.machines))
        # obs space
        self.observation_space = spaces.Dict({
                "age": spaces.Box(low=0., high=self.max_duration, shape=(len(self.machines),), dtype=np.float32),
                "condition": spaces.MultiBinary(len(self.machines)),
                "resources": spaces.Box(low=0., high=self.max_resource, shape=(len(self.machines),), dtype=np.float32),
                "failure_prob": spaces.Box(low=0., high=1., shape=(len(self.machines),), dtype=np.float32)
                })

    def reset(self):

        self.timer = time.time()
        self.machine_a = Machine(output_rate=1, param=10)
        self.machine_b = Machine(output_rate=2, param=15)
        self.machines = [self.machine_a, self.machine_b]

        return self.observation()

    def observation(self):

        state = {
            "age": [],
            "condition": [],
            "resources": [],
            "failure_prob": []
        }

        for machine in self.machines:
            state['age'].append(machine.age)
            state['condition'].append(machine.working)
            state['resources'].append(machine.capacity)
            state['failure_prob'].append(machine.fail_prob)

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
            reward -= machine.repair_cost * machine.repair_status
            if machine.working == False:
                reward -= 120

        return reward

    def check_done(self):

        if np.sum([machine.capacity for machine in self.machines]) == 0:
            done = True
        else:
            done = False

        return done

    def step(self, action):

        now = time.time()

        for machine in self.machines:
            # Deterioriation
            machine.failure_sampling(now)
            # Inventory 
            machine.update_inv(now)
            # Reset Repair Status
            machine.repair_status = 0

        # Interactions
        if action == 0:
            self.machine_a.repair(now)
        if action == 1:
            self.machine_b.repair(now)
        if action == 2:
            self.machine_a.repair(now)
            self.machine_b.repair(now)
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
            result['time'] = int(time.time() - self.timer)
            result['repair_count'] = [machine.repair_counter for machine in self.machines]
            result['reward'] = self.get_reward()
            
            clear_output(wait=True)
            display(result)
            time.sleep(1)