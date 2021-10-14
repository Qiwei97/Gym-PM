# Import Modules
import numpy as np
from reliability.Distributions import Weibull_Distribution

class Machine:

    def __init__(self, output_rate, alpha, repair_cost=30, resupply_cost=5, product_price=3):

        self.capacity = 100
        self.working = 1

        # Cost elements
        self.repair_cost = repair_cost
        self.resupply_cost = resupply_cost
        self.product_price = product_price
        self.output_rate = output_rate

        self.age = 0
        self.repair_counter = 0
        self.repair_time = 0
        self.repair_status = 0

        # Survival
        self.reliability_dist = Weibull_Distribution(alpha=alpha, beta=5)
        self.survival_prob = 1
        self.ttf = np.round(self.reliability_dist.random_samples(1)) # Time to failure
    
    # Resource Usage
    def update_inv(self):

        if self.capacity == 0:
            return

        if self.working:
            self.capacity -= self.output_rate

    # Deterioration
    def failure_check(self):

        # Break down
        if self.age >= self.ttf:
            self.working = 0
            return
        
        if self.working and self.capacity > 0:
            self.age += 1
            self.survival_prob = self.reliability_dist.SF(self.age)
            
    def repair(self):

        # Random fault types
        self.repair_time = np.random.randint(low=2, high=5)
        
        self.ttf = np.round(self.reliability_dist.random_samples(1)) # Time to failure
        self.repair_counter += 1
        self.working = 1
        self.age = 0
        self.repair_status = 1