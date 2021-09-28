# Import Modules
import time
import numpy as np

class Machine:

    def __init__(self, output_rate, param, repair_cost=30, resupply_cost=5, product_price=3):

        self.capacity = 25
        self.working = 1
        self.age = 0
        self.fail_prob = 0
        self.repair_counter = 0
        self.repair_status = 0

        start = time.time()
        self.inv_time = start
        self.det_time = start
        self.ref_age = start

        # cost elements
        self.repair_cost = repair_cost
        self.resupply_cost = resupply_cost
        self.downtime = 0
        self.output = 0
        self.product_price = product_price
        self.output_rate = output_rate
        self.repair_time = 0

        # weibull
        self.param = param
    
    # Resource Usage
    def update_inv(self, now):

        if self.capacity == 0:
            return

        if self.working:
            interval = now - self.inv_time
            if interval >= 1: # Update every second
                self.capacity -= self.output_rate
                self.output += self.output_rate
                self.inv_time = now

    # Deterioration
    def failure_sampling(self, now, sampling_time=1, b=5):

        def weibull(a, b, x):
            return 1 - np.e ** -((x / a) ** b)
        
        interval = now - self.det_time
        if interval >= sampling_time:
            self.det_time = now
            if self.working and self.capacity > 0:
                self.age = now - self.ref_age
                self.fail_prob = weibull(self.param, b, self.age)
                self.working = 1 - np.random.binomial(1, self.fail_prob)
            else:
                self.downtime += sampling_time # if dont repair immediately + stock out time
             
    def repair(self, now):

        # Random fault types
        self.repair_time = np.random.randint(low=2, high=5)
        
        self.downtime += self.repair_time
        self.repair_counter += 1
        self.working = 1
        self.ref_age = now
        self.age = 0
        self.repair_status = 1