# Import Modules
import numpy as np
from gym_pm.utils import load_data
from reliability.Distributions import Weibull_Distribution

class Train:

    def __init__(self, alpha, repair_cost=30):

        self.working = 1

        # Cost elements
        self.repair_cost = repair_cost

        self.age = 0
        self.repair_counter = 0
        self.repair_time = 0
        self.repair_status = 0

        # Survival
        self.reliability_dist = Weibull_Distribution(alpha=alpha, beta=5)
        self.ttf = np.round(self.reliability_dist.random_samples(1)) # Time to failure

    # Deterioration
    def failure_check(self):

        # Break down
        if self.age >= self.ttf:
            self.working = 0
            return
        
        if self.working:
            self.age += 1
            
    def repair(self):

        self.ttf = np.round(self.reliability_dist.random_samples(1)) # Time to failure
        self.age = 0
        
        self.repair_time = 1
        self.repair_counter += 1
        self.working = 1
        self.repair_status = 1


class Train_v2:

    def __init__(self, data, repair_cost=30):

        self.working = 1

        # Cost elements
        self.repair_cost = repair_cost

        self.repair_counter = 0
        self.repair_time = 0
        self.repair_status = 0

        # Data
        self.df = load_data(data)

    # Deterioration
    def failure_check(self, time_step):

        # Break down
        if self.df.iloc[time_step].Failure:
            self.working = 0
            return
        
    def repair(self):

        self.repair_time = 1
        self.repair_counter += 1
        self.working = 1
        self.repair_status = 1


class Factory:

    def __init__(self, output_rate, alpha, episode_length,
                 repair_cost=30, resupply_cost=3, 
                 storage_cost=2, resupply_qty=10, 
                 lead_time=3, product_price=75):

        self.capacity = 20
        self.working = 1
        self.output = 0

        # Cost elements
        self.repair_cost = repair_cost
        self.resupply_cost = resupply_cost
        self.storage_cost = storage_cost
        self.output_rate = output_rate
        self.resupply_qty = resupply_qty
        self.lead_time = lead_time
        self.product_price = product_price

        self.age = 0
        self.repair_counter = 0
        self.repair_time = 0
        self.repair_status = 0
        self.resupply_status = 0

        # List of Lead Time for Countdown
        self.resupply_list = np.array([])

        # Survival
        self.reliability_dist = Weibull_Distribution(alpha=alpha, beta=5)
        self.survival_prob = 1
        self.ttf = np.round(self.reliability_dist.random_samples(1)) # Time to failure

        # Demand
        self.demand_dist = np.random.poisson(2, episode_length)
        self.fulfilled_orders = 0
    
    # Resource Usage
    def update_inv(self):

        if self.capacity < self.output_rate:
            return

        if self.working:
            self.capacity -= self.output_rate
            self.output += self.output_rate

    # Deterioration
    def failure_check(self):

        # Break down
        if self.age >= self.ttf:
            self.working = 0
            return
        
        if self.working and self.capacity > 0:
            self.age += 1
            self.survival_prob = self.reliability_dist.SF(self.age)

    # Update Lead time 
    def update_LT(self):

        if len(self.resupply_list) > 0:
            self.resupply_list -= 1 # Count down
            # Replenish Stock
            self.capacity += self.resupply_qty * sum(self.resupply_list <= 0)
            self.resupply_list = self.resupply_list[self.resupply_list > 0]

    def repair(self):

        # Random fault types
        self.repair_time = np.random.randint(low=2, high=5)
        
        self.ttf = np.round(self.reliability_dist.random_samples(1)) # Time to failure
        self.repair_counter += 1
        self.working = 1
        self.age = 0
        self.repair_status = 1

    def resupply(self):

        # Send resupply orders
        self.resupply_list = np.append(self.resupply_list, self.lead_time)
        self.resupply_status = 1

