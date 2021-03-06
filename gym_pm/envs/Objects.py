# Import Modules
import numpy as np
import pandas as pd
from gym_pm.utils import load_data
from reliability.Distributions import Weibull_Distribution
from timeseries_generator.external_factors import CountryGdpFactor, EUIndustryProductFactor
from timeseries_generator import Generator, HolidayFactor, SinusoidalFactor, WeekdayFactor, WhiteNoise

class Train:

    def __init__(self, alpha, 
                 beta, repair_cost):

        self.working = 1

        # Cost elements
        self.repair_cost = repair_cost

        self.age = 0
        self.repair_counter = 0
        self.repair_time = 0
        self.repair_status = 0

        # Survival
        self.reliability_dist = Weibull_Distribution(alpha=alpha, beta=beta)
        self.ttf = np.round(self.reliability_dist.random_samples(1)) # Time to failure

    # Deterioration
    def failure_check(self):

        if self.working:
            self.age += 1

        # Break down
        if self.age >= self.ttf:
            self.working = 0
            return
            
    def repair(self):

        self.ttf = np.round(self.reliability_dist.random_samples(1)) # Time to failure
        self.age = 0
        
        self.repair_time = 1
        self.repair_counter += 1
        self.working = 1
        self.repair_status = 1


class Train_v2:

    def __init__(self, data, split, 
                 repair_cost, file_path):

        self.working = 1

        # Cost elements
        self.repair_cost = repair_cost

        self.repair_counter = 0
        self.repair_time = 0
        self.repair_status = 0

        # Data
        try:
            self.df = load_data(file_path, data, split)
        except:
            print('Data not found. Please run the create_data function.')

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

    def __init__(self, output_rate, alpha, 
                 beta, capacity, episode_length,
                 repair_cost, resupply_cost, 
                 storage_cost, resupply_qty, 
                 lead_time, product_price):

        self.working = 1
        self.output = 0

        # Cost elements
        self.output_rate = output_rate
        self.capacity = capacity
        self.repair_cost = repair_cost
        self.resupply_cost = resupply_cost
        self.storage_cost = storage_cost
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
        self.reliability_dist = Weibull_Distribution(alpha=alpha, beta=beta)
        self.ttf = np.round(self.reliability_dist.random_samples(1)) # Time to failure

        # Demand
        self.demand_dist = Generator(factors = {CountryGdpFactor(),
                                                EUIndustryProductFactor(),
                                                HolidayFactor(holiday_factor = 1.5),
                                                WeekdayFactor(
                                                    factor_values = {4: 1.05, 5: 1.15, 6: 1.15}  
                                                ), # Here we assign a factor of 1.05 to Friday, and 1.15 to Sat/Sun
                                                SinusoidalFactor(wavelength = 365, 
                                                                amplitude = 0.2, 
                                                                phase = 365/4, 
                                                                mean = 1),
                                                WhiteNoise(stdev_factor = 0.3)},
                                    features = {"country": ["Netherlands"],
                                                "store": ["store1"],
                                                "product": ["winter jacket"]},
                                    date_range = pd.date_range(start = '2010', periods = (episode_length + 1), freq = 'D'),
                                    base_value = 4).generate()
               
        self.demand_dist[self.demand_dist.value < 0] = 0
        self.demand_dist = round(self.demand_dist['value']).values
        self.fulfilled_orders = 0
    
    # Resource Usage
    def update_inv(self):

        if self.capacity < self.output_rate:
            return

        if self.working and self.repair_status == 0:
            self.capacity -= self.output_rate
            self.output += self.output_rate

    # Deterioration
    def failure_check(self):
        
        if self.working and self.capacity >= self.output_rate:
            self.age += 1

        # Break down
        if self.age >= self.ttf:
            self.working = 0
            return

    # Update Lead time 
    def update_lt(self):

        if len(self.resupply_list) > 0:
            self.resupply_list -= 1 # Count down lead time
            # Replenish Stock
            self.capacity += self.resupply_qty * sum(self.resupply_list <= 0)
            self.resupply_list = self.resupply_list[self.resupply_list > 0]

    def repair(self):
        
        self.ttf = np.round(self.reliability_dist.random_samples(1)) # Time to failure
        self.age = 0

        self.repair_time = 1
        self.repair_counter += 1
        self.working = 1
        self.repair_status = 1

    def resupply(self):

        # Send resupply orders
        self.resupply_list = np.append(self.resupply_list, self.lead_time)
        self.resupply_status = 1


class Factory_v2:

    def __init__(self, output_rate, data, 
                 split, capacity, repair_cost, 
                 resupply_cost, storage_cost, 
                 resupply_qty, lead_time, 
                 product_price, file_path):

        self.working = 1
        self.output = 0

        # Cost elements
        self.output_rate = output_rate
        self.capacity = capacity
        self.repair_cost = repair_cost
        self.resupply_cost = resupply_cost
        self.storage_cost = storage_cost
        self.resupply_qty = resupply_qty
        self.lead_time = lead_time
        self.product_price = product_price

        self.repair_counter = 0
        self.repair_time = 0
        self.repair_status = 0
        self.resupply_status = 0

        # List of Lead Time for Countdown
        self.resupply_list = np.array([])

        # Data
        try:
            self.df = load_data(file_path, data, split)
        except:
            print('Data not found. Please run the create_data function.')

        # Demand
        self.demand_dist = Generator(factors = {CountryGdpFactor(),
                                                EUIndustryProductFactor(),
                                                HolidayFactor(holiday_factor = 1.5),
                                                WeekdayFactor(
                                                    factor_values = {4: 1.05, 5: 1.15, 6: 1.15}  
                                                ), # Here we assign a factor of 1.05 to Friday, and 1.15 to Sat/Sun
                                                SinusoidalFactor(wavelength = 365, 
                                                                amplitude = 0.2, 
                                                                phase = 365/4, 
                                                                mean = 1),
                                                WhiteNoise(stdev_factor = 0.3)},
                                    features = {"country": ["Netherlands"],
                                                "store": ["store1"],
                                                "product": ["winter jacket"]},
                                    date_range = pd.date_range(start = '2010', periods = (len(self.df) + 1), freq = 'D'),
                                    base_value = 4).generate()
               
        self.demand_dist[self.demand_dist.value < 0] = 0
        self.demand_dist = round(self.demand_dist['value']).values
        self.fulfilled_orders = 0
    
    # Resource Usage
    def update_inv(self):

        if self.capacity < self.output_rate:
            return

        if self.working and self.repair_status == 0:
            self.capacity -= self.output_rate
            self.output += self.output_rate

    # Deterioration
    def failure_check(self, time_step):

        # Break down
        if self.df.iloc[time_step].Failure:
            self.working = 0
            return

    # Update Lead time 
    def update_lt(self):

        if len(self.resupply_list) > 0:
            self.resupply_list -= 1 # Count down lead time
            # Replenish Stock
            self.capacity += self.resupply_qty * sum(self.resupply_list <= 0)
            self.resupply_list = self.resupply_list[self.resupply_list > 0]

    def repair(self):

        self.repair_time = 1
        self.repair_counter += 1
        self.working = 1
        self.repair_status = 1

    def resupply(self):

        # Send resupply orders
        self.resupply_list = np.append(self.resupply_list, self.lead_time)
        self.resupply_status = 1
