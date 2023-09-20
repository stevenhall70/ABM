from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from geopy.distance import geodesic
import pandas as pd
import numpy as np
import os
import json
import random
import uuid
import matplotlib.pyplot as plt
from scipy.spatial import distance
import logging
# Ensure the Logs directory exists
log_dir = 'Logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# Delete all logs that exist in the Logs directory
for file in os.listdir(log_dir):
    file_path = os.path.join(log_dir, file)
    if os.path.isfile(file_path):
        os.remove(file_path)
# Set up basic logging configuration
log_path = os.path.join(log_dir, 'potato_simulation.log')
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s -%(name)s - %(levelname)s - %(message)s',
                    filename=log_path,
                    filemode='w')
# Set the logging level for matplotlib to WARNING to suppress DEBUG and INFO logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)
########################################################################################################################
# Global Functions
########################################################################################################################
Seasonality_factors = {
        1: 1,
        2: 1,
        3: 1,
        4: 0.9553,
        5: 0.9553,
        6: 0.9553,
        7: 1.01,
        8: 1.01,
        9: 1.01,
        10:1.16,
        11:1.16,
        12:1.16,
    }

def get_distance(model, loc1, loc2):
    distance = model.distances.get((loc1, loc2), model.distances.get((loc2, loc1)))
    if distance is None:
        distance = geodesic(loc1, loc2).miles
        model.distances[(loc1, loc2)] = distance  # storing the computed distance for future use
    return distance


def compute_all_distances(model):
    distances = {}
    
    # From Potato Production Areas to Storage facilities
    for ppa in model.production_areas:
        for storage in model.storage_agents:
            dist = geodesic((ppa.latitude, ppa.longitude), (storage.location[0], storage.location[1])).miles
            distances[(ppa.unique_id, storage.unique_id)] = dist

    # From Storage facilities to Logistics
    for storage in model.storage_agents:
        for logistics in model.logistics_agents:
            dist = geodesic((storage.location[0], storage.location[1]), (logistics.location[0], logistics.location[1])).miles
            distances[(storage.unique_id, logistics.unique_id)] = dist
    
    # From Logistics to Retailers
    for logistics in model.logistics_agents:
        for retailer in model.retailer_agents:
            dist = geodesic((logistics.location[0], logistics.location[1]), (retailer.location[0], retailer.location[1])).miles
            distances[(logistics.unique_id, retailer.unique_id)] = dist

    # From Consumers to Retailers
    for consumer in model.consumer_agents:
        for retailer in model.retailer_agents:
            dist = geodesic((consumer.location[0], consumer.location[1]), (retailer.location[0], retailer.location[1])).miles
            distances[(consumer.unique_id, retailer.unique_id)] = dist

    # From Production Areas to Raw Water Agents
    for ppa in model.production_areas:
        for raw_water_agent in model.raw_water_agents:
            dist = geodesic((ppa.latitude, ppa.longitude), (raw_water_agent.location[0], raw_water_agent.location[1])).miles
            distances[(ppa.unique_id, raw_water_agent.unique_id)] = dist

    # Nearest Neighbors to a given Production Area (for up to 10 nearest neighbors)
    for target_ppa in model.production_areas:
        distances_to_neighbors = [(geodesic((target_ppa.latitude, target_ppa.longitude), 
                                             (neighbor.latitude, neighbor.longitude)).miles, neighbor.unique_id) 
                                  for neighbor in model.production_areas if neighbor != target_ppa]
        
        # Storing distances for the 10 nearest neighbors
        nearest_neighbors = sorted(distances_to_neighbors, key=lambda x: x[0])[:10]
        distances[(target_ppa.unique_id, 'nearest_neighbors')] = nearest_neighbors

    # Storing the computed distances in the model
    model.distances = distances


def compute_total_weight(model):
    total_weight = 0
    for area in model.schedule.agents:
        if isinstance(area, PotatoProductionArea):
            total_weight += area.compute_total_weight()
    return total_weight
########################################################################################################################
# Agent Classes
########################################################################################################################
### Base Agent Class ###
class BaseAgent:
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
##################################################
#Potato Production Area Agent Subclass
class PotatoProductionArea(BaseAgent):
    #potato_weight_lbs = 0.2
    #conversion_factor_cwt = potato_weight_lbs / 100
    desired_cwt_per_acre = 20
    weight_per_agent_in_cwt = 1000
    first_potato_logged = False
    @property
    def acres_per_potato_agent(self):
        desired_cwt_per_acre = PotatoProductionArea.desired_cwt_per_acre
        weight_per_agent_in_cwt = PotatoProductionArea.weight_per_agent_in_cwt
        return weight_per_agent_in_cwt / desired_cwt_per_acre
    
    def __init__(self, unique_id, model, acres, name, owner, state, latitude, longitude, raw_water_agent):
        super().__init__(unique_id, model)
        self.name = name
        self.acres = acres
        self.potatoes = []
        self.owner = owner
        self.state = state
        self.latitude = latitude
        self.longitude = longitude
        self.potatoes_planted = False
        self.agent_type = 'PotatoProductionArea'
        self.raw_water_agent = raw_water_agent
        self.potato_count = 0
        self.total_water_demand = 0
        self.total_weight = 0
        self.dead_potatoes_weight = 0
        self.water_consumed = 0
        self.potatoes_ready_for_harvest = 0
        self.replant_day = random.randint(330,390)
        
        print(f"Production Area Name: {self.name}, Acres: {self.acres}")

    def pre_plant_irrigation(self):
        field_capacity = 0.18  
        desired_moisture_fraction = random.uniform(0.7, 0.8)
        current_moisture_fraction = random.uniform(0.2,0.8)  
        soil_depth = 1  
        required_moisture_fraction = desired_moisture_fraction - current_moisture_fraction
        water_needs_acre_foot = self.acres * required_moisture_fraction * field_capacity * soil_depth
        water_needs_gallons = water_needs_acre_foot * 325851
            
        return water_needs_gallons
    
    def plant_potatoes(self):
        # Calculating the total number of agents needed for the production area
        total_agents_for_production_area = int(self.acres * self.desired_cwt_per_acre / self.weight_per_agent_in_cwt)
        
        for agent_num in range(total_agents_for_production_area):
            # Setting the seed weight in cwt directly using the class attribute
            potato = Potato(unique_id=f"{self.unique_id}_{self.potato_count}", 
                        model=self.model, 
                        initial_water_needs=0,  
                        seed_weight=self.weight_per_agent_in_cwt,  
                        production_area=self)
            
            # Log only for the first potato 
            if agent_num == 0 and not PotatoProductionArea.first_potato_logged:
                potato.is_first = True
                potato.setup_logger()
                potato.tracked_logger.debug(f'For Potato {potato.unique_id}, seed_weight_cwt is {self.weight_per_agent_in_cwt}.')
                PotatoProductionArea.first_potato_logged = True
            
            self.potatoes.append(potato)
            self.model.schedule.add(potato)
            self.potato_count += 1
            
        print(f"Planted {len(self.potatoes)} potato agents (each representing {PotatoProductionArea.weight_per_agent_in_cwt} cwt) in {self.name} owned by {self.owner}.")

    def average_plants_per_acre(self):
        return self.potato_count / self.acres
    def notify_water_demand(self, potato, amount):
        self.total_water_demand += amount
    
    def notify_weight_increase(self, weight_increase):
        """Adjusts the total weight in the production area when a potato grows."""
        self.total_weight += weight_increase

    def adjust_weight_after_death(self, weight_lost):
        """Adjusts the total weight in the production area when a potato dies."""
        self.total_weight = max(0, self.total_weight - weight_lost)
        self.dead_potatoes_weight += weight_lost

    def consume_water(self, water_amount):
        self.water_consumed += water_amount
    
    def potato_ready_for_harvest(self, potato):
        self.potatoes_ready_for_harvest += 1
        potato.status = 'Storage'
        self.call_for_harvest(potato)

    def find_nearest_storage(self):
        # Fetch the distances from model.distances
        distances = {storage: self.model.distances[(self.unique_id, storage.unique_id)] for storage in self.model.storage_agents if storage.current_volume < storage.capacity}
    
        # return the storage facilities sorted by distance
        return sorted(distances.keys(), key=lambda x: distances[x])


    def call_for_harvest(self, potato):
        volume_to_harvest = potato.weight * 1000  # Convert to cwt, assuming potato.weight is in 1000 cwt units
        while volume_to_harvest > 0:
            storages = self.find_nearest_storage()
            if storages:
                for storage in storages:
                    space_in_storage = storage.capacity - storage.current_volume
                    if space_in_storage >= volume_to_harvest:
                        storage.current_volume += volume_to_harvest
                        self.total_weight -= max(0,volume_to_harvest / 1000)  # Reduce the weight from PPA and convert back to 1000 cwt units
                        volume_to_harvest = 0
                        break
                    else:
                        volume_to_harvest -= space_in_storage
                        storage.current_volume = storage.capacity  # fill up the storage
                        self.total_weight -= max(0,space_in_storage / 1000)  # Reduce the weight from PPA and convert back to 1000 cwt units
            else:
                # If there's no storage facility available, send to 'Export'
                self.model.export_bin_volume += volume_to_harvest
                volume_to_harvest = 0
        self.model.schedule.remove(potato)

    def replant(self):
        # Reset necessary attributes for a new planting season
        self.potatoes_planted = False
        #self.potato_count = 0
        self.total_water_demand = 0
        self.total_weight = 0
        self.dead_potatoes_weight = 0
        self.water_consumed = 0
        self.potatoes_ready_for_harvest = 0
        self.potatoes = []
        self.replant_day = random.randint(330, 390) 
        print(f"Replanting in {self.name}.")

    

    def step(self):
        if not self.potatoes_planted:
            # Request water for pre-planting irrigation
            pre_plant_water_needed = self.pre_plant_irrigation()
            water_received_for_preplant = self.raw_water_agent.supply_water(pre_plant_water_needed)
            
            # Check if received water is less than what's needed
            if water_received_for_preplant < pre_plant_water_needed:
                print(f"Not enough water for pre-plant irrigation in {self.name}. Skipping planting this step.")
                return  # Exit the current step
            
            # Plant potatoes after receiving sufficient water for pre-plant irrigation
            self.plant_potatoes()
            self.potatoes_planted = True
            
        # After all potatoes have made their requests, fetch water
        water_received = self.raw_water_agent.supply_water(self.total_water_demand)
        total_water_needs = sum([potato.water_needs for potato in self.potatoes])
        
        for potato in self.potatoes:
            if total_water_needs == 0:  # To avoid division by zero
                water_allocated = water_received / len(self.potatoes)
            else:
                water_allocated = (potato.water_needs / total_water_needs) * water_received
            potato.receive_water(water_allocated)
            self.consume_water(water_allocated)
        self.total_weight = sum([potato.weight for potato in self.potatoes if potato.status == 'Growing'])

        if self.model.schedule.time == self.replant_day:
           self.replant()
################################################
###Potato Agent Subclass###
class Potato(BaseAgent):
  
    GROWTH_STAGES = ["Germination", "Vegetative", "Tubering", "Maturation", "Harvest Ready"]
    AREA_PER_POTATO_SQFT = 1
    DROUGHT_THRESHOLD = 7  # Days without sufficient water before dying
    DEFAULT_GROWTH_RATE = 0.05
    EPSILON = 1e-10  # Small constant to prevent division by zero
    def __init__(self, unique_id, model, seed_weight, production_area, growth_rate=DEFAULT_GROWTH_RATE, initial_water_needs=0):
        super().__init__(unique_id, model)
        self.weight = seed_weight
        growth_multiplier = np.random.normal(loc=16149,scale=10)
        self.growth_stage = 0
        self.water_needs = initial_water_needs
        self.days_in_current_stage = 0
        self.alive = True
        self.agent_type = 'Potato'
        self.current_water_supply = 0
        self.time_stages = self.generate_time_stages()  # Generate the random time stages
        #self.time_stages.append(14)  # Adding 14 days for the final stage
        self.max_weight = self.weight * growth_multiplier
        self.growth_rate = growth_rate  # Now using the parameterized growth rate
        self.week_in_stage = 0
        self.days_without_sufficient_water = 0
        self.production_area = production_area  # Storing the reference to the associated PotatoProductionArea
        self.logger = logging.getLogger('Potato')
        self.first_potato_id = None
        self.is_first = False
        self.harvestable = False
        self.status = 'Growing'
    
    def setup_logger(self):
        if self.is_first:
            self.tracked_logger = logging.getLogger(f"TrackedPotato_{self.unique_id}")
            self.tracked_logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(f'Logs/tracked_{self.unique_id}.log', mode='w')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.tracked_logger.addHandler(fh)

    def water_needs_for_stage(self, growth_stage):
        """ Returns the daily water needs based on the growth stage and week in that stage in gallons. """
        
        # Step 1: Calculate water needs in inches/day based on growth stage
        if growth_stage == 0:  # Germination
            water_needs_inches = 0  # No additional irrigation after initial pre-plant irrigation
        elif growth_stage == 1:  # Vegetative
            # Start with 0.071 inches/day and increase by 0.071 inches/day every week until 0.214 inches/day
            water_needs_inches = min(0.214, 0.071 + (self.week_in_stage * 0.071))
        elif growth_stage == 2:  # Tubering
            water_needs_inches = 0.286
        elif growth_stage == 3:  # Maturation
            water_needs_inches = 0.321 * (0.9 ** self.week_in_stage)
        elif growth_stage == 4:  # Harvest Ready
            water_needs_inches = 0.00  
        else:
            raise ValueError("Invalid growth stage provided.")
        # Step 2: Convert the water needs from inches/acre/day to gallons/acre/day
        gallons_per_acre_per_day = water_needs_inches * 325829  
        # Step 3: Determine total gallons/day for the area represented by the agent
        acres_represented_by_agent = self.production_area.acres_per_potato_agent
        total_gallons_per_day = gallons_per_acre_per_day * acres_represented_by_agent
        
        if hasattr(self, 'is_first') and self.is_first:
            self.tracked_logger.debug(
                f'Potato {self.unique_id} at growth stage {self.GROWTH_STAGES[self.growth_stage]} (week {self.week_in_stage}) '
                f'calculated water needs as: '
                f'\n - {water_needs_inches} inches/day '
                f'\n - {gallons_per_acre_per_day} gallons/acre/day '
                f'\n - Representing {acres_represented_by_agent} acres '
                f'\n - Total of {total_gallons_per_day} gallons/day '
                f'on step {self.model.schedule.time}.'
            )
        return total_gallons_per_day
        
    def generate_time_stages(self):
        # Define the mean and stddev for each stage (values can be adjusted)
        stage_parameters = [(22.5, 7.5), (22.5, 7.5), (37.5, 22.5), (22.5, 7.5)]
        
        time_stages = [int(np.random.normal(mean, stddev)) for mean, stddev in stage_parameters]
        
        # Constrain the values within the defined limits for each stage
        time_stages = [min(30, max(15, time_stages[0])), #Germination
                    min(30, max(15, time_stages[1])),   #Vegetation
                    min(60, max(15, time_stages[2])), #Tubering
                    min(30, max(15, time_stages[3]))] #Maturation
        time_stages.append(14) # Adding 14 days for skin hardening in the ground before harvest
    
        return time_stages
    
    def calculate_growth(self):
        # Logistic growth function implementation
        t_max = sum(self.time_stages)  # Total time to reach maximum growth
        L = self.max_weight  # Carrying capacity or final weight
        k = self.growth_rate * (self.current_water_supply / (self.water_needs + self.EPSILON))  # Growth rate adjusted by water supply, prevent division by zero
        t = sum(self.time_stages[:self.growth_stage]) + self.days_in_current_stage  # Current time in the simulation
        if hasattr(self, 'is_first') and self.is_first:
            self.logger.debug(f"Values before calculating growth: k={k}, t={t}, L={L}, current_weight={self.weight}")
        self.weight = L / (1 + (L - self.weight) / self.weight * np.exp(-k * t))
        if hasattr(self, 'is_first') and self.is_first:
            self.tracked_logger.debug(f'Potato {self.unique_id} grew to weight {self.weight} on step {self.model.schedule.time}.')

    def time_to_next_stage(self, growth_stage):
        return self.time_stages[growth_stage]
    
    def receive_water(self, water_amount):
        self.logger.debug(f'Received {water_amount} units of water.')
        self.current_water_supply += water_amount
        # Logging received water and new water supply
        if hasattr(self, 'is_first') and self.is_first:
            self.logger.debug(f'Potato {self.unique_id} has current water supply of {self.current_water_supply} units on step {self.model.schedule.time}.')

    def transition_growth_stage(self):
        """Move to the next growth stage if the duration exceeds the defined time for the current stage."""
            # Check if the agent's time in the current growth stage exceeds the specified time for that stage
        while self.days_in_current_stage >= self.time_stages[self.growth_stage]:
            
            # If there's a next growth stage, transition to it
            if self.growth_stage < len(self.GROWTH_STAGES) - 1:
                self.growth_stage += 1
                self.days_in_current_stage = 0
                self.week_in_stage = 0
                    
                # Log the transition if this potato is the first one (for logging purposes)
                if hasattr(self, 'is_first') and self.is_first:
                    self.tracked_logger.info(f'Potato {self.unique_id} transitioned to {self.GROWTH_STAGES[self.growth_stage]} stage on step {self.model.schedule.time}.')
                
            # If there's no next growth stage, break out of the loop to prevent an infinite loop
            else:
                break
        # Check for the "Harvest Ready" stage:
        if self.growth_stage == 4 and self.days_in_current_stage == 14:
            self.production_area.potato_ready_for_harvest(self)

    def die(self):
        self.alive = False
        # Notify the PotatoProductionArea of the death so it can adjust its weight calculations directly here
        self.production_area.adjust_weight_after_death(self.weight)
        if hasattr(self, 'is_first') and self.is_first:
            self.tracked_logger.error(f'Potato {self.unique_id} died due to lack of water on step {self.model.schedule.time}.')

    def notify_potato_death(self, weight_lost):
        # Assuming you have a reference to the PotatoProductionArea agent. Adjust as needed.
        self.production_area.adjust_weight_after_death(weight_lost)

    def step(self):
        if not self.alive:
            return
        # If the potato is ready for harvest, it shouldn't grow or request water.
        if not self.harvestable:
            # Inform the PotatoProductionArea of the water need
            self.water_needs = self.water_needs_for_stage(self.growth_stage)
            self.production_area.notify_water_demand(self, self.water_needs)
            
            # Calculate water needs
            self.current_water_supply -= self.water_needs
            self.current_water_supply = max(0, self.current_water_supply)  # Ensure the water doesn't go negative
            # Check if the potato is getting enough water
            if self.current_water_supply < self.water_needs:
                self.days_without_sufficient_water += 1
                if hasattr(self, 'is_first') and self.is_first:
                    self.tracked_logger.warning(f'Potato {self.unique_id} has not received enough water for {self.days_without_sufficient_water} days on step {self.model.schedule.time}.')
            else:
                self.days_without_sufficient_water = 0  # Reset the counter if the potato gets enough water
            # Introduce a drought stress or death mechanism
            # If a potato doesn't receive sufficient water for 7 days in a row, it dies.
            if self.days_without_sufficient_water >= self.DROUGHT_THRESHOLD:
                self.die()
                return
            # Adjust water needs every week (assuming a week is 7 steps) for specific growth stages.
            if self.model.schedule.time % 7 == 0 and self.growth_stage in [1, 3]:
                self.water_needs *= 1.05  # Placeholder: Increase by 5% every week. Adjust based on real-world dynamics.
            # Calculate growth at the beginning of each step
            self.calculate_growth()
        # Check if it's time to transition to a new growth stage
        self.days_in_current_stage += 1
        # Increment week counter
        if self.days_in_current_stage % 7 == 0:
            self.week_in_stage += 1
        self.transition_growth_stage()  
##############################################
#Raw Water Agent Subclass
class RawWater(BaseAgent):
    def __init__(self, unique_id, model, name, water_capacity=float('inf'), facility_type=None, location=None, disruption_percentage=0):
        super().__init__(unique_id, model)
        self.name = name
        self.facility_type = facility_type
        self.water_capacity = water_capacity  # Only using water capacity now
        self.disruption_percentage = disruption_percentage
        self.status = "Normal"
        self.location = location
        self.agent_type = 'Raw Water'
        self.water_use_1 = 0
        self.disruption_days_left = 0

        

    def supply_water(self, demand):
        """
        Supplies water based on the demand. Returns the amount of water actually supplied which can be less than the demand if capacity is limited.
        """
        # Update water capacity based on the model step
        if self.model.schedule.time == 365 and self.model.drought_present:
            self.water_capacity = self.water_use_1
        elif self.model.schedule.time == 365:
            self.water_capacity = float('inf')
                     
        # Calculate the actual water supplied (could be less than demand due to total capacity)
        supplied_water = min(demand, self.water_capacity)
        
        # Reduce the current water capacity
        self.water_capacity -= supplied_water

        # If in the first year, track the water usage
        if self.model.schedule.time < 365:
            self.water_use_1 += supplied_water

        # Logging
        logging.debug(f"{self.name} - Water Requested: {demand}, Water Supplied: {supplied_water}, Remaining Capacity: {self.water_capacity}")
        return supplied_water
    
    # Placeholder for potential future methods
    def drought_disruption(self, percentage):
        """
        Introduces a disruption in the form of reducing the available water capacity.
        """
        self.water_capacity *= (1 - percentage / 100)

    def full_disruption(self, days):
        """
        Fully disrupts the water supply for a given number of days.
        """
        self.water_capacity = 0
        self.disruption_days_left = days

    def restore_water_capacity(self):
        if self.model.drought_present:
            self.water_capacity = self.water_capacity
        else:
            self.water_capacity = float('inf')

    def step(self):
        """
        Steps the RawWater agent. This could include events like natural replenishment of water.
        """
        if self.model.schedule.time == 400 and self.model.drought_present:  # 400 days is roughly June
            self.drought_disruption(30)  # Disrupting the water availability by x%

        # Handle ongoing disruptions
        if self.model.enable_random_disruptions and self.disruption_days_left > 0:
            self.disruption_days_left -= 1

            # If the disruption ends, restore the water capacity (or however you intend to reset it)
            if self.disruption_days_left == 0:
                self.restore_water_capacity()

################################################
# Storage Facility Agent Subclass
class Storage(BaseAgent):
    def __init__(self, unique_id, model,name, location, capacity):
        super().__init__(unique_id, model)
        self.name = name
        self.location = location
        self.capacity = capacity
        self.current_volume = 0
        self.shrinkage_rate = .0003
        self.has_notified_logistics = False
        self.month_counter = 0
        self.agent_type = "Storage"

    def apply_shrinkage(self):
        self.current_volume = self.current_volume * (1 - self.shrinkage_rate)

    def notify_logistics(self):
        if self.current_volume >0 and not self.has_notified_logistics:
            for agent in self.model.schedule.agents:
                if isinstance(agent, Logistics):
                    agent.update_storage_inventory(self.unique_id, self.current_volume)
                    print(f"Storage Agent {self.unique_id} notified Logistics Agent {agent.unique_id} of inventory: {self.current_volume} units.")
            self.has_notified_logistics = True
    
    def step(self):
        self.apply_shrinkage()
        self.notify_logistics()

################################################
#Logistic Agent Subclass
class Logistics(BaseAgent):
    def __init__(self, unique_id, model, name, location):
        super().__init__(unique_id, model)
        self.name = name
        self.location = location
        self.storage_inventories = {}
        #self.storage_ids = [storage.unique_id for storage in self.storage_agents]#AttributeError: 'Logistics' object has no attribute 'storage_agents'
        self.received_bids = []
        self.commodity_price = 0.0725 #commodity price paid to farmers
        self.commodity_markup_factor = 1.6 #Markup factor for the commodity price
        self.shipping_markup_factor = 1.2 #Markup factor the shipping costs
        self.has_notified_retailers = False
        self.total_profit = 0
        self.ask_price = 0
        

    def update_storage_inventory(self, storage_id, current_volume):
        self.storage_inventories[storage_id] = current_volume
        if current_volume > 0:
            self.has_notified_retailers = False

    def wholesale_price(self):
        self.base_price_per_unit = self.commodity_price * self.commodity_markup_factor
        return self.base_price_per_unit

    def notify_retailers(self):
        if sum(self.storage_inventories.values()) > 0 and not self.has_notified_retailers:
            price = self.wholesale_price()
            for agent in self.model.schedule.agents:
                if isinstance(agent, Retailer):
                    agent.receive_order_notification(self.unique_id, price)
            self.has_notified_retailers = True

    def find_nearest_storage(self, location):
        # Assuming location is a tuple containing the unique_id of the retailer and the coordinates
        retailer_id, coords = location

        # Retrieve the list of distances from this retailer to all storages
        storage_distances = [(storage_id, self.model.distances.get((retailer_id, storage_id), float('inf'))) 
                             for storage_id in self.model.storage_ids]

        if storage_distances:
            # Find the storage agent with the minimum distance to this retailer
            nearest_storage_id, _ = min(storage_distances, key=lambda x: x[1])

            # Find and return the actual Storage agent corresponding to nearest_storage_id
            nearest_storage = next((agent for agent in self.model.schedule.agents if agent.unique_id == nearest_storage_id), None)
            return nearest_storage
        else:
            return None
        
    def find_nearest_shipper_to_storage(self, storage_agent):
        shippers = [agent for agent in self.model.schedule.agents if isinstance(agent, Shipper)]
        if shippers:
            # Using precomputed distances from the model's distances dictionary
            shipper_distances = [(shipper, self.model.distances.get((shipper.unique_id, storage_agent.unique_id), float('inf'))) 
                                for shipper in shippers]
            
            nearest_shipper, _ = min(shipper_distances, key=lambda x: x[1])
            return nearest_shipper
        else:
            return None

    
    def calculate_total_cost(self, volume_in_pounds, total_shipping_cost):
        # Here, implement logic to calculate the total cost including the product and shipping cost
        base_price_per_unit = self.commodity_price * self.commodity_markup_factor
        product_cost = volume_in_pounds * base_price_per_unit
        return product_cost + total_shipping_cost

    def receive_bid(self, retailer_id, price_per_pound, volume_in_pounds):
        # Logic to handle the bid, e.g., adding it to a list of received bids
        self.received_bids.append({'retailer_id': retailer_id, 'price_per_pound': price_per_pound, 'volume_in_pounds': volume_in_pounds})

    def propose_counteroffer(self, retailer, new_price_per_pound, volume_in_pounds):
        # Notify the retailer of the counteroffer
        retailer.receive_counteroffer(self.unique_id, new_price_per_pound, volume_in_pounds)
        self.received_bids.append({'price_per_pound': new_price_per_pound}) # Corrected the attribute name
        print(f"Proposed counteroffer to retailer {retailer.unique_id} for {volume_in_pounds} pounds at {new_price_per_pound} per pound.")
    
    def accepted_counteroffer(self, retailer, volume_in_pounds, price_per_pound, storage_id):
        # Logic to handle the acceptance of a counteroffer and initiate the shipping process
        self.update_storage_volume(storage_id, volume_in_pounds)  # You will need to determine the appropriate storage_id here

        # Now initiate the shipping process
        self.initiate_shipping(retailer, volume_in_pounds, price_per_pound, storage_id)

    def update_storage_volume(self, storage_id, volume_decrement):
        # Finding the correct storage agent using the passed storage_id and updating its volume
        storage_agent = next(agent for agent in self.model.schedule.agents if isinstance(agent, Storage) and agent.unique_id == storage_id)
        storage_agent.current_volume -= volume_decrement
    
    def evaluate_bids(self):
        total_inventory = sum(self.storage_inventories.values())
        if total_inventory <= 0 or not self.received_bids:
            return

        self.received_bids.sort(key=lambda x: (-x['price_per_pound'], -x['volume_in_pounds']))

        for bid in self.received_bids:
            retailer_id = bid['retailer_id']
            retailer = next(agent for agent in self.model.schedule.agents if agent.unique_id == retailer_id and isinstance(agent, Retailer))
            
            # Step 1: Find the nearest storage agent to the retailer
            nearest_storage = self.find_nearest_storage(retailer.location)
            if nearest_storage:
                bid['nearest_storage'] = nearest_storage
            else:
                continue  # or handle this situation in a different way
            
            # Step 2: Get the nearest shipper to the nearest storage agent
            nearest_shipper = self.find_nearest_shipper_to_storage(nearest_storage.location)

            if not nearest_shipper:
                continue  # or handle this situation in a different way

            # Now get the distance from the nearest Shipper to the Retailer from the global distances dictionary
            distance_to_retailer_from_shipper = self.model.distances.get((nearest_shipper.unique_id, retailer.unique_id), float('inf'))

            
            # Assuming Shipper has a class variable called shipping_cost_per_mile
            total_shipping_cost = (distance_to_retailer_from_shipper * Shipper.shipping_cost_per_mile) * self.shipping_markup_factor

            # Determine the total cost for this bid, including the wholesale cost of goods and the marked-up shipping cost
            total_cost = self.calculate_total_cost(bid['volume_in_pounds'], total_shipping_cost)
            
            total_bid_price = bid['volume_in_pounds'] * bid['price_per_pound']

            if total_bid_price >= total_cost and bid['volume_in_pounds'] <= total_inventory:
                # Accept the bid
                total_inventory -= bid['volume_in_pounds']
                self.update_storage_volume(bid['nearest_storage'].unique_id, bid['volume_in_pounds'])
    
                print(f"Accepted bid from retailer {retailer.unique_id} for {bid['volume_in_pounds']} pounds at {bid['price_per_pound']} per pound. Shipping cost: {total_shipping_cost}")
                self.total_profit += total_bid_price - total_cost
                self.ask_price = total_cost

                # Initiate shipping
                self.initiate_shipping(retailer, bid['volume_in_pounds'], bid['price_per_pound'], bid['nearest_storage'].unique_id)
            else:
                # Propose a counteroffer or reject the bid
                new_price_per_pound = total_cost / bid['volume_in_pounds']
                self.propose_counteroffer(retailer, new_price_per_pound, bid['volume_in_pounds'])

    def initiate_shipping(self, retailer, volume_in_pounds, price_per_pound, storage_id):
        nearest_shipper = self.find_nearest_shipper_to_storage(self.model.get_agent_by_id(storage_id).location)
        
        if nearest_shipper:
            distance_to_retailer_from_shipper = self.get_distance(nearest_shipper.location, retailer.location)
            estimated_delivery_time = distance_to_retailer_from_shipper / 45  # Assuming speed is 45 mph
            
            # Notify the shipper to initiate the delivery
            nearest_shipper.initiate_delivery(retailer.unique_id, volume_in_pounds, estimated_delivery_time, price_per_pound)



    def step(self):
        self.notify_retailers()
        self.evaluate_bids()
        self.received_bids = []

################################################
#Shipper Agent Subclass
class Shipper(BaseAgent):
    def __init__(self, unique_id, model, name, refrigeratedtrucks, drivers, location):
        super().__init__(unique_id, model)
        self.name = name
        self.trucks = {i: 0 for i in range(refrigeratedtrucks)}  # 0 indicates that the truck is available at time 0
        self.drivers = {i: 0 for i in range(drivers)}  # 0 indicates that the driver is available at time 0
        self.location = location
        self.shipping_cost_per_mile = 0.10
        self.order_queue = []  # To store the orders

    def initiate_delivery(self, retailer_id, volume_in_pounds, estimated_delivery_time,price_per_pound):
        # Add the order to the queue
        self.order_queue.append({
            'retailer_id': retailer_id,
            'volume_in_pounds': volume_in_pounds,
            'estimated_delivery_time': estimated_delivery_time,
            'price_per_pound': price_per_pound
        })

    def find_available_truck(self):
        for truck_id, time_available in self.trucks.items():
            if time_available <= self.model.schedule.steps:
                return truck_id
        return None

    def find_available_driver(self):
        for driver_id, time_available in self.drivers.items():
            if time_available <= self.model.schedule.steps:
                return driver_id
        return None

    def assign_truck_and_driver_to_order(self, order, truck, driver):
        # Further logic to mark the order as assigned and potentially notify other agents or update the order's status
        delivery_time = order['estimated_delivery_time']
        self.trucks[truck] = self.model.schedule.steps + delivery_time
        self.drivers[driver] = self.model.schedule.steps + delivery_time

    def complete_delivery(self, retailer_id, volume_in_pounds, price_per_pound, distance_to_retailer_from_shipper):
        retailer = self.model.get_agent_by_id(retailer_id)  
        retailer.receive_order(volume_in_pounds, price_per_pound)

        # Calculate the time taken for the round trip at an average speed of 45 mph, using the passed distance
        total_round_trip_distance = distance_to_retailer_from_shipper * 2
        average_speed = 45.0  # in miles per hour
        total_round_trip_time = (total_round_trip_distance / average_speed) 

        # Update the truck and driver's next available time based on the round trip time
        next_available_time = self.model.schedule.steps + total_round_trip_time
        print(f"Shipper {self.unique_id} completed delivery to retailer {retailer_id}. Round trip time: {total_round_trip_time} hours. Next available time: {next_available_time}")

    def assign_orders(self):
        for order in self.order_queue:
            available_truck = self.find_available_truck()
            available_driver = self.find_available_driver()

            if available_truck is not None and available_driver is not None:
                self.assign_truck_and_driver_to_order(order, available_truck, available_driver)
                self.complete_delivery(order['retailer_id'], order['volume_in_pounds'], None, order['estimated_delivery_time'])
                self.order_queue.remove(order)
                break  # Break to avoid modifying the list while iterating

    def step(self):
        # Try to assign orders at each step
        self.assign_orders()

        
################################################
# Retailer Agent Subclass
class Retailer(BaseAgent):
    def __init__(self, unique_id, model, name, location, address, city, owner):
        super().__init__(unique_id, model)
        self.name = name
        self.location = location
        self.address = address
        self.city = city
        self.owner = owner
        self.inventory = 0
        self.total_demand = 0
        self.total_cost = 0
        self.demand_reports = {}
        self.bid_price_history = []
        self.price_per_unit_weight = 0.6 # initial price estimation, this could be adjusted based on historical data
        self.markup_percentage = 1.3 # initial markup percentage, can also be adjusted dynamically
        self.consumer_price = 0

    def find_nearest_logistics(self):
        logistics_agents = [agent for agent in self.model.schedule.agents if isinstance(agent, Logistics)]
        if logistics_agents:
            # Using precomputed distances from the model's distances dictionary
            logistics_distances = [(log_agent, self.model.distances.get((self.unique_id, log_agent.unique_id), float('inf'))) 
                                for log_agent in logistics_agents]

            nearest_logistics_agent, _ = min(logistics_distances, key=lambda x: x[1])
            return nearest_logistics_agent
        else:
            return None

        
    def calculate_bid_price(self):
        # Retrieve the current month
        current_month = ((self.model.schedule.steps // 4) + 4) % 12
        if current_month == 0:
            current_month = 12

        # Retrieve the seasonality factor for the current month
        seasonality_factor = Seasonality_factors[current_month]

        # Step 1: Estimation of Consumer Willingness to Pay (now using seasonality)
        estimated_consumer_willingness_to_pay = self.price_per_unit_weight * seasonality_factor

        # Step 2: Markup Strategy
        markup_percentage = self.markup_percentage * seasonality_factor

        # Calculate the initial bid price considering the markup and seasonality factor
        initial_bid_price = estimated_consumer_willingness_to_pay * (1 + markup_percentage)
        self.bid_price_history.append(initial_bid_price)

        # Step 4 to 6: Dynamic Bids (Subsequent Steps)
        # Logic to adjust the bid price based on historical transaction data, 
        # supply and demand dynamics, and other factors.
        
        # Might adjust the markup percentage based on recent supply and demand trends:
        # if supply > demand:
        #     self.markup_percentage = ... (reduce markup percentage)
        # else:
        #     self.markup_percentage = ... (increase markup percentage)

        return initial_bid_price
    
    def receive_consumer_demand(self, consumer_id, average_weekly_demand):
        # Here you can add code to update the retailer's total demand with the demand from this consumer
        self.total_demand += average_weekly_demand
        #print(f"Received average weekly demand of {average_weekly_demand} from consumer {consumer_id}")

    def receive_order_notification(self, logistics_id, price):
        #print(f"Retailer {self.unique_id} received order notification from Logistics {logistics_id} with price {price}")
        order_quantity = self.decide_order_quantity(price)

    def decide_order_quantity(self, price):
        if price >0:
            order_quantity = self.total_demand / price
            self.inventory += order_quantity
            return order_quantity
        else:
            return 0

    def place_bid(self):
        nearest_logistics_agent = self.find_nearest_logistics()
        if nearest_logistics_agent:
            bid_price_per_pound = self.calculate_bid_price()
            bid_volume_in_pounds = self.total_demand  # Adjust as necessary

            # Notify the nearest logistics agent of the bid
            nearest_logistics_agent.receive_bid(self.unique_id, bid_price_per_pound, bid_volume_in_pounds)
    
    def receive_counteroffer(self, logistics_id, new_price_per_pound, volume_in_pounds):
        # Logic to handle receiving a counteroffer, e.g., decide whether to accept or reject the counteroffer
        logistics_agent = next(agent for agent in self.model.schedule.agents if agent.unique_id == logistics_id and isinstance(agent, Logistics))

        if self.inventory == 0:
            # If the inventory is zero, accept any counteroffer
            self.inventory += volume_in_pounds
            self.bid_price_history.append(new_price_per_pound)
            self.current_purchase_cost = new_price_per_pound * volume_in_pounds  # Track the cost of the current purchase

            logistics_agent.accepted_counteroffer(self.unique_id, volume_in_pounds, new_price_per_pound)  # Notify the logistics agent of the acceptance

            print(f"Accepted counteroffer from logistics {logistics_id} for {volume_in_pounds} pounds at {new_price_per_pound} per pound.")
        else:
            # Here, you might add other conditions for accepting or rejecting the counteroffer based on other factors such as the price offered, remaining demand, etc.
            # For example:
            if new_price_per_pound <= self.calculate_bid_price():
                self.inventory += volume_in_pounds
                self.total_demand -= volume_in_pounds
                self.bid_price_history.append(new_price_per_pound)
                self.current_purchase_cost = new_price_per_pound * volume_in_pounds

                logistics_agent.accepted_counteroffer(self.unique_id, volume_in_pounds, new_price_per_pound)  # Notify the logistics agent of the acceptance

                print(f"Accepted counteroffer from logistics {logistics_id} for {volume_in_pounds} pounds at {new_price_per_pound} per pound.")
            else:
                print(f"Rejected counteroffer from logistics {logistics_id}.")


    def receive_order(self, volume_in_pounds, price_per_pound):
        self.inventory += volume_in_pounds
        self.total_cost += volume_in_pounds * price_per_pound  # Assuming you have a total_cost attribute to track the total cost of potatoes purchased
        self.bid_price_history.append(price_per_pound)
        
        # Here, you might also add code to calculate the profit based on the markup and update the financial records
        # You could also implement logic to adjust the markup percentage based on recent transaction data
        # ...
        self.bid_price = price_per_pound
        print(f"Retailer {self.unique_id} received an order of {volume_in_pounds} pounds at {price_per_pound} per pound. New inventory level: {self.inventory}")

    def calculate_consumer_price(self):
        return self.price_per_unit_weight * self.markup_percentage

    def step(self):
        self.place_bid()

################################################
# Consumer Agent Subclass
class Consumer(BaseAgent):
    def __init__(self, unique_id, model, population, income, location):
        super().__init__(unique_id, model)
        self.population = population
        self.location = location
        self.income = income
        self.start_reporting_step = 90
        self.step_counter = -90
        self.total_demand_over_30_steps = 0
        self.total_demand = 0  # New attribute to track cumulative demand
        self.potato_demand_in_weight = 0  # Initialize this attribute to avoid potential errors
        self.price_per_unit_weight = 0.95  # Average cost of a lb of potatoes

    def estimate_potato_demand(self, price_per_unit_weight):
        current_month = ((self.model.schedule.steps // 4) + 4) % 12
        if current_month == 0:
            current_month = 12
        
        seasonality_factor = Seasonality_factors[current_month]

        weekly_income = (self.income * self.population * (1 - 0.30)) / 52
        grocery_spending = weekly_income * 0.113
        potato_spending = grocery_spending * 0.01
        self.potato_demand_in_weight = potato_spending / price_per_unit_weight * seasonality_factor
        self.total_demand_over_30_steps += self.potato_demand_in_weight
        self.total_demand += self.potato_demand_in_weight  # Add the demand to the cumulative total

    def notify_retailer_of_demand(self):
        nearest_retailer = self.find_nearest_retailer()
        if nearest_retailer:
            average_weekly_demand = self.total_demand_over_30_steps / 30
            nearest_retailer.receive_consumer_demand(self.unique_id, average_weekly_demand)
            self.total_demand_over_30_steps = 0
    
    def find_nearest_retailer(self):
        retailers = [agent for agent in self.model.schedule.agents if isinstance(agent, Retailer)]
        if retailers:
            # Using precomputed distances from the model's distances dictionary
            retailer_distances = [(retailer, self.model.distances.get((self.unique_id, retailer.unique_id), float('inf'))) 
                                for retailer in retailers]

            nearest_retailer, _ = min(retailer_distances, key=lambda x: x[1])
            return nearest_retailer
        else:
            return None


    def step(self):
        if self.model.schedule.steps >= self.start_reporting_step:
            self.estimate_potato_demand(self.price_per_unit_weight)
            self.step_counter += 1
            if self.step_counter == 0:
                self.notify_retailer_of_demand()
                self.step_counter = 0

##################################################################################################
### Main Model Class ###
######################################################################################################
class PotatoSupplyChain(Model):
    def __init__(self, runtime=None, enable_random_disruptions = False, drought_present=False):
        np.random.seed(None)
        self.schedule = RandomActivation(self)
        self.running = True
        self.runtime = runtime
        self.distances = {}
        self.export_bin_volume = 0
        self.drought_present = drought_present
        self.disruption_schedule = []
        self.enable_random_disruptions = enable_random_disruptions
        self.affected_ppas = {}
           
        # Initialize agents
        self.initialize_raw_water_agents()
        self.initialize_production_areas()
        self.initialize_storage_agents()
        self.intialize_logistics_agents()
        self.initialize_shipper_agents()
        self.initialize_consumer_agents()
        self.initialize_retailer_agents()

        compute_all_distances(self)

        self.datacollector = DataCollector(
    model_reporters={
        "Total_Weight_cwt": lambda m: m.compute_global_total_weight() / 10000,
        "Weights_by_Production_Area_cwt": lambda m: {area.unique_id: area.total_weight / 10000 for area in m.production_areas},
        "Dead_Potatoes_Weight_cwt": lambda m: sum(area.dead_potatoes_weight for area in m.production_areas) / 10000,
        "Water_Consumption": lambda m: sum(area.water_consumed for area in m.production_areas)/1000000,
        "Total_Local_Storage": lambda m: m.compute_total_local_storage()/10000,
        "Total_Exports": lambda m: m.compute_total_exports()/10000,
        "Affected_PPAs": lambda m: m.affected_ppas,
        
    },
    agent_reporters={
        "AskPrice": lambda a: a.ask_price if isinstance(a, Logistics) else None,
        "Logistics Agent Profit": lambda a: a.total_profit if isinstance(a, Logistics) else None,
        #"BidPrice": lambda a: a.bid_price if isinstance(a, Retailer) else None,
        #"ConsumerPrice": lambda a: a.consumer_price if isinstance(a, Retailer) else None,
        "Consumer_Demand": lambda agent: sum(a.total_demand for a in agent.model.schedule.agents if isinstance(a, Consumer)), 
        "Retailer_Order_Volume": lambda agent: sum(a.total_demand for a in agent.model.schedule.agents if isinstance(a, Retailer)) 
    }
)

        
    def initialize_raw_water_agents(self):
        with open('./AHAJSON/rawWater.json') as file:
            raw_water_data = json.load(file)
        self.raw_water_agents = [
            RawWater(unique_id=item['id'], model=self, name=item['name'],
                     facility_type=item['facilityTypeName'],
                     location=(item['latitude'], item['longitude'])) for item in raw_water_data
        ]

    def initialize_production_areas(self):
        with open('./AHAJSON/potatoProductionAreas.json') as file:
            data = json.load(file)
        total_acres = 328858
        num_agents = len(data)
        base_acres_per_agent = total_acres // num_agents
        remaining_acres = total_acres - base_acres_per_agent * num_agents
        self.production_areas = []
        for item in data:
            extra_acres = random.randint(0, remaining_acres // 2)
            remaining_acres -= extra_acres
            acres_assigned = base_acres_per_agent + extra_acres
            closest_water_agent = self.find_closest_water_agent(item['latitude'], item['longitude'], self.raw_water_agents)
            agent = PotatoProductionArea(
                unique_id=item['id'],
                model=self, 
                acres=acres_assigned,
                name=item['name'],
                owner=item['owner'],
                state=item['state'],
                latitude=item['latitude'],
                longitude=item['longitude'],
                raw_water_agent=closest_water_agent
            )
            self.production_areas.append(agent)
            self.schedule.add(agent)
        if remaining_acres > 0:
            random_agent = random.choice(self.production_areas)
            random_agent.acres += remaining_acres
        print(f"Production Areas: {len(self.production_areas)}")
        active_raw_water_agents = set(area.raw_water_agent for area in self.production_areas)
        for agent in active_raw_water_agents:
            self.schedule.add(agent)
        print(f"Raw Water Agent Count: {len(self.raw_water_agents)}")

    def initialize_storage_agents(self):
        with open('./AHAJSON/StorageFacilities.json') as file:
            storage_facility_data = json.load(file)
        self.storage_agents = [
            Storage(unique_id=item['id'], model=self, name=item['name'],
                    location=(item['latitude'], item['longitude']),
                    capacity=item['specificProperties']['storagecapacitycwt']) for item in storage_facility_data
        ]
        print(f"Storage Agents Count: {len(self.storage_agents)}")

        for agent in self.storage_agents:
            self.schedule.add(agent)

    def intialize_logistics_agents(self):
        with open('./AHAJSON/logistics.json') as file:
            logistics_data = json.load(file)
        self.logistics_agents = [
            Logistics(unique_id=item['id'], model=self, name=item['name'],
                    location=(item['latitude'], item['longitude'])) for item in logistics_data
        ]
        print(f"Logistics Agents Count: {len(self.logistics_agents)}")
        for agent in self.logistics_agents:
            self.schedule.add(agent)


    def initialize_shipper_agents(self):
        with open('./AHAJSON/shippers.json') as file:
            shipper_data = json.load(file)
        self.shipper_agents = [
            Shipper(unique_id = item['id'], model=self, name=item['name'], refrigeratedtrucks=item['specificProperties']['refrigeratedtrucks'], drivers=item['specificProperties']['drivers'],
                    location=(item['latitude'], item['longitude'])) for item in shipper_data
        ]
        print(f"Shipper Agents Count: {len(self.shipper_agents)}")

        for agent in self.shipper_agents:
            self.schedule.add(agent)

    def initialize_consumer_agents(self):
        with open('./AHAJSON/consumerAgents.json') as file:
            consumer_data = json.load(file)
        self.consumer_agents = [
            Consumer(unique_id=uuid.uuid4(), model=self,  population=item['Population'], income = item['Income'],
            location = (item['latitude'], item['longitude'])) for item in consumer_data
        ]
        print(f"Consumer Agents Count: {len(self.consumer_agents)}")
        for agent in self.consumer_agents:
            self.schedule.add(agent)

    def initialize_retailer_agents(self):
        with open('./AHAJSON/retailers.json') as file:
            retailer_data = json.load(file)
        self.retailer_agents = [
            Retailer(unique_id=item['id'], model=self, name=item['name'],
            address = item['address'], city = item['city'], owner= item['owner'],
            location = (item['latitude'], item['longitude'])) for item in retailer_data
        ]
        print(f"Retailer Agents Count: {len(self.retailer_agents)}")
        for agent in self.retailer_agents:
            self.schedule.add(agent)

    def compute_total_local_storage(self):
        return sum(storage.current_volume for storage in self.storage_agents)

    def compute_total_exports(self):
        return self.export_bin_volume

    def compute_global_total_weight(self):
        return sum(area.total_weight for area in self.production_areas)
    
    def find_closest_water_agent(self, latitude, longitude, raw_water_agents):
        closest_distance = float('inf')
        closest_water_agent = None
        for water_agent in raw_water_agents:
            loc1 = (latitude, longitude)
            loc2 = water_agent.location
            dist = get_distance(self, loc1, loc2)
            if dist < closest_distance:
                closest_distance = dist
                closest_water_agent = water_agent
        return closest_water_agent

    
    def generate_random_disruptions(self):
        if self.schedule.time == 365:  # Start of the second year
            for _ in range(10):  # Generate 10 random disruptions-cyber attacks for example
                start_day = random.randint(400, 600)  # Random day in the second year
                duration = random.randint(3, 10)  # Random duration of disruption

                # Select a random PPA
                target_PPA = random.choice(self.production_areas)  

                # Identify the nearest neighbors to the target PPA
                nearest_neighbors = self.get_nearest_neighbors(target_PPA)

                # Create a disruption event and add it to a schedule
                self.disruption_schedule.append({
                    'start_day': start_day,
                    'end_day': start_day + duration,
                    'target_PPA': target_PPA,
                    'nearest_neighbors': nearest_neighbors
                })

    def get_nearest_neighbors(self, target_PPA):
        # Use the 'nearest_neighbors' key in self.distances to get the 10 nearest neighbors
        nearest_neighbors_ids = self.distances.get((target_PPA.unique_id, 'nearest_neighbors'), [])
        
        # Get the neighbor objects from the IDs
        nearest_neighbors = [neighbor for neighbor in self.production_areas if neighbor.unique_id in [item[1] for item in nearest_neighbors_ids]]
        
        return nearest_neighbors

    def execute_disruptions(self):
        current_day = self.schedule.time
        for disruption in self.disruption_schedule:
            if disruption['start_day'] <= current_day <= disruption['end_day']:
                # Apply the disruption to the target PPA's associated RawWater agent and its nearest neighbors
                disruption['target_PPA'].raw_water_agent.full_disruption(disruption['end_day'] - current_day)
                for neighbor in disruption['nearest_neighbors']:
                    neighbor.raw_water_agent.full_disruption(disruption['end_day'] - current_day)
                # Updating the affected PPAs property
                disruption_days = disruption['end_day'] - current_day
                affected_ppas_today = [(disruption['target_PPA'].unique_id, disruption_days)] + [(neighbor.unique_id, disruption_days) for neighbor in disruption['nearest_neighbors']]
                
                self.affected_ppas[current_day] = affected_ppas_today

                # Print the information about the disruption
                affected_ppa_ids = [ppa[0] for ppa in affected_ppas_today]
                print(f"On day {current_day}, the following PPAs were affected by a disruption for {disruption_days} days: {', '.join(affected_ppa_ids)}")

        
    def step(self):
        self.datacollector.collect(self)
        if self.enable_random_disruptions:
            self.generate_random_disruptions()
            self.execute_disruptions()
        self.schedule.step()        
#################################################################################################################
#Run the Model
##############################################################################################################

number_of_runs = 1  # Setup for Monte Carlo simulations
results = []

for i in range(number_of_runs):
    np.random.seed(i)  # Setting a different seed for each run
    
    model = PotatoSupplyChain(enable_random_disruptions=False, drought_present=False)

    for _ in range(180):  
        if model.running:
            model.step()
        else:
            break

    model_data = model.datacollector.get_model_vars_dataframe()
    model_data.to_json(f'./AHAJSON/Model_output_run_{i}.json')  # Saving data with a unique filename for each run
    
    agent_data = model.datacollector.get_agent_vars_dataframe().reset_index()
    agent_data.to_json(f'./AHAJSON/Agent_output_run_{i}.json')

    first_year_max = model_data.iloc[0:365]['Total_Weight_cwt'].max()
    second_year_max = model_data.iloc[365:730]['Total_Weight_cwt'].max()
    
    lost_production = round((first_year_max - second_year_max) * 100,0)
    lost_production = '{:,.2f}'.format(lost_production)
    
    results.append({
        "run": i,
        "first_year_max": first_year_max,
        "second_year_max": second_year_max,
        "lost_production": lost_production
    })

    print(f"Run {i}:")
    print(f"The highest total weight during the first year is: {first_year_max} cwt.")
    print(f"The highest total weight during the second year is: {second_year_max} cwt.")
    print(f"The lost production is: {lost_production} pounds.")
    print("----------")


####################################################################################################################
# Plot Graphs
# ################################################################################################################## 
# 1. Total Weight of Potatoes Over Time
model_data['Total_Weight_cwt'].plot()
plt.title('Total Weight of Potatoes Over Time')
plt.xlabel('Time (steps)')
plt.ylabel('Total Weight (10000 cwt)')
plt.show()
# 2. Weights by Individual Production Areas Over Time
weights_by_area = model_data['Weights_by_Production_Area_cwt'].apply(pd.Series)
weights_by_area.plot(legend=False)  # Setting legend to False to avoid overcrowding, adjust as needed
plt.title('Weights by Individual Production Areas Over Time')
plt.xlabel('Time (steps)')
plt.ylabel('Weight (10000 cwt)')
plt.show()
# 3. Dead Potatoes Weight Over Time
model_data['Dead_Potatoes_Weight_cwt'].plot()
plt.title('Total Dead Potatoes Weight Over Time')
plt.xlabel('Time (steps)')
plt.ylabel('Dead Potatoes Weight (10000 cwt)')
plt.show()
# 4. Water Consumption Over Time
model_data['Water_Consumption'].plot()
plt.title('Water Consumption Over Time')
plt.xlabel('Time (steps)')
plt.ylabel('Water Consumed (Million Gallons)')
plt.show()
# 5. Storage
model_data["Total_Local_Storage"].plot()
plt.title('Local Storage Capacity')
plt.xlabel('Time (steps)')
plt.ylabel('Potatoes Stored Locally  (10000 cwt)')
plt.show()
# 6. Exports
model_data["Total_Exports"].plot()
plt.title('Exports')
plt.xlabel('Time (steps)')
plt.ylabel('Potatoes Exported (10000 cwt)')
plt.show()
#7 Consumer Demand
agent_data["Consumer_Demand"].plot()
plt.title('Consumer Demand')
plt.xlabel('Time (steps)')
plt.ylabel('Consumer Demand')
plt.show()
            
#8 Retail Orders
agent_data["Retailer_Order_Volume"].plot()
plt.title('Retailer Orders')
plt.xlabel('Time (steps)')
plt.ylabel('Orders')
plt.show()

#9 Prices
plt.figure(figsize=(10, 6))

# Plotting the various prices over time
agent_data["Logistics_Agent_Profit"].plot(label='Logistics Agent Profit', color='blue')
agent_data["Ask_Price"].plot(label='Ask Price', color='green')
agent_data["Bid_Price"].plot(label='Bid Price', color='red')
agent_data["Consumer_Price"].plot(label='Consumer Price', color='purple')

# Adding title and labels
plt.title('Price Dynamics Over Time')
plt.xlabel('Time (steps)')
plt.ylabel('Price ($)')

# Adding legend
plt.legend()

# Displaying the plot
plt.show()
'''
############################################################################################################################
#MESA Visualization
############################################################################################################################
area_ids = [area.unique for area in model.production_areas]
#Error: area_ids = [area.unique for area in model.production_areas]
AttributeError: 'PotatoProductionArea' object has no attribute 'unique'
# Generate color list
num_areas = len(area_ids)
colormap = plt.cm.tab20
colors = [colormap[1] for i in range(num_areas)]
#Convert to hex codes
hex_colors = [plt.colors.rgb2hex(color) for color in colors]
chart_total_weight = ChartModule([{"Label": "Total_Weight_1000cwt", "Color": "Green"}],
                                  data_collector_name='datacollector')
chart_weight_by_area = ChartModule([{"Label": area_id, "Color": color} for area_id, color in zip(area_ids, hex_colors)],
                                  data_collector_name='datacollector')
chart_dead_potatoes = ChartModule([{"Label": "Dead_Potatoes_Weight_1000cwt", "Color": "Red"}],
                                  data_collector_name='datacollector')
chart_water_consumption = ChartModule([{"Label": "Water_Consumption", "Color": "Blue"}],
                                  data_collector_name='datacollector')
# When you initiate your server, add these ChartModules
server = ModularServer(PotatoSupplyChain, [chart_total_weight, chart_weight_by_area, chart_dead_potatoes, chart_water_consumption], "Potato Supply Chain", {})
'''
