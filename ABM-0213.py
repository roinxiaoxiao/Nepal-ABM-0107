import numpy as np
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import Slider, Choice
import random
import pandas as pd

# Constant definitions
EDUCATION_LEVELS = {
    "none": 0,
    "primary": 1,
    "secondary": 2,
    "higher": 3
}

INCOME_BY_EDUCATION = {
    "none": 500,
    "primary": 800,
    "secondary": 1200,
    "higher": 2000
}

class HouseholdAgent(Agent):
    """Household agent class"""
    def __init__(self, unique_id: int, model: Model):
        super().__init__(unique_id, model)
        # Basic attributes
        self.education_level = random.choice(list(EDUCATION_LEVELS.keys()))
        self.income = INCOME_BY_EDUCATION[self.education_level] * random.uniform(0.8, 1.2)
        self.pos = None
        
        # Migration-related attributes
        self.has_migrant = False
        self.remittance = 0
        self.migration_cost = random.uniform(2000, 4000)  # Adding migration cost
        self.social_network = random.uniform(0, 1)
        
        # Child marriage risk-related attributes
        self.child_marriage_risk = self.calculate_initial_risk()
        self.cultural_norm = random.uniform(0, 1)
        
        # Spillover effects
        self.knowledge_spillover = 0
        self.economic_spillover = 0
        self.norm_spillover = 0
        
    def calculate_initial_risk(self):
        """Calculate initial child marriage risk"""
        base_risk = {
            "none": random.uniform(0.6, 0.8),
            "primary": random.uniform(0.4, 0.6),
            "secondary": random.uniform(0.2, 0.4),
            "higher": random.uniform(0.1, 0.2)
        }
        return base_risk[self.education_level]
    
    def calculate_poverty_level(self):
        """Calculate poverty level"""
        return max(0, 1 - (self.income / 2000))  # 2000 as the poverty threshold
    
    def calculate_environmental_risk(self):
        """Calculate environmental risk"""
        return random.uniform(0, 1)  # Simplified implementation
    
    def calculate_network_effect(self):
        """Calculate social network effect"""
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True)
        if not neighbors:
            return self.social_network
        
        migrant_neighbors = sum(1 for n in neighbors if 
                              isinstance(n, HouseholdAgent) and n.has_migrant)
        return migrant_neighbors / len(neighbors) if neighbors else 0
    
    def calculate_remittance_buffer(self):
        """Calculate remittance buffer effect"""
        return (self.economic_spillover * 0.4 + 
                self.knowledge_spillover * 0.3 + 
                self.norm_spillover * 0.3)
    
    def migration_decision(self):
        """Implement migration decision"""
        income_effect = self.income + self.remittance
        network_effect = self.calculate_network_effect()
        env_risk = self.calculate_environmental_risk()
        
        prob = 1 / (1 + np.exp(-(
            -3.0 +  # More negative base value
            -0.0005 * income_effect +  # Reduced income impact
            0.5 * network_effect +  # Reduced network effect
            0.3 * env_risk +  # Reduced environmental risk impact
            -0.001 * self.migration_cost  # Increased cost impact
        )))
        
        return random.random() < prob
    
    def child_marriage_decision(self):
        """Implement child marriage decision"""
        poverty = self.calculate_poverty_level()
        env_stress = self.calculate_environmental_risk()
        remittance_buffer = self.calculate_remittance_buffer()
        
        prob = 1 / (1 + np.exp(-(
            -2.0 +  # alpha_0
            2.0 * poverty +  # alpha_1
            1.0 * self.cultural_norm +  # alpha_2
            1.0 * env_stress +  # alpha_3
            -1.0 * remittance_buffer  # alpha_4
        )))
        
        return random.random() < prob
    
    def step(self):
        """Agent step function"""
        self.calculate_spillover_effects()
        
        # Migration decision
        if not self.has_migrant and self.migration_decision():
            self.has_migrant = True
            self.remittance = random.uniform(500, 2000)  # Simplified remittance simulation
        
        # Child marriage decision
        if self.child_marriage_decision():
            self.child_marriage_risk = min(1, self.child_marriage_risk * 1.2)
        else:
            self.child_marriage_risk = max(0, self.child_marriage_risk * 0.8)
