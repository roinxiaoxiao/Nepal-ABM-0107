import numpy as np
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import NetworkModule, ChartModule
from mesa.visualization.UserParam import Slider
import networkx as nx
import random
from typing import List, Dict, Any


# Ethnicity parameter settings
ETHNICITY_PARAMS = {
    "Bramin": {"income_base": 1500, "migration_tendency": 0.3, "color": "#E41A1C"},
    "Hill": {"income_base": 1200, "migration_tendency": 0.25, "color": "#377EB8"},
    "Dalit": {"income_base": 800, "migration_tendency": 0.2, "color": "#4DAF4A"},
    "Newar": {"income_base": 1800, "migration_tendency": 0.35, "color": "#984EA3"},
    "Terai": {"income_base": 1000, "migration_tendency": 0.25, "color": "#FF7F00"},
    "Others": {"income_base": 1200, "migration_tendency": 0.25, "color": "#A65628"}
}

class HouseholdAgent(Agent):
    """Household agent"""
    def __init__(self, unique_id: int, model: Model, ethnicity: str, location: tuple):
        super().__init__(unique_id, model)
        self.ethnicity = ethnicity
        self.location = location
        self.pos = unique_id
        
        # Initialize basic attributes
        params = ETHNICITY_PARAMS[ethnicity]
        self.income = random.uniform(params["income_base"] * 0.8, 
                                   params["income_base"] * 1.2)
        
        # Agricultural attributes
        self.land_area = random.uniform(0.35, 400)
        self.agricultural_investment = 0
        
        # Migration-related attributes
        self.migrants: List[int] = []
        self.total_remittance = 0
        self.previous_remittance = 0
        
        # Development indicators
        self.education_investment = 0
        self.child_marriage_risk = random.uniform(0.4, 0.8)
        
    def update_development_indicators(self):
        """Update development indicators"""
        if self.total_remittance > 0:
            # Education investment
            self.education_investment = min(self.total_remittance * 0.3, 100000)
            
            # Update child marriage risk
            remittance_effect = -0.05 * (self.total_remittance / 50000)
            education_effect = -0.1 * (self.education_investment / 10000)
            
            self.child_marriage_risk = max(0.1, min(0.9, 
                                        self.child_marriage_risk + remittance_effect + education_effect))
    
    def step(self):
        """Step update"""
        self.update_development_indicators()
        self.previous_remittance = self.total_remittance

class MigrantAgent(Agent):
    """Migrant agent"""
    def __init__(self, unique_id: int, model: Model, home_id: int, current_location: str):
        super().__init__(unique_id, model)
        self.home_id = home_id
        self.current_location = current_location
        self.is_international = "F" in current_location
        self.remittance_ability = random.uniform(0, 9875000)
        
    def calculate_remittance(self) -> float:
        """Calculate remittance amount"""
        base_amount = self.remittance_ability * random.uniform(0.8, 1.2)
        if self.is_international:
            base_amount *= 1.5
        return min(base_amount, 9875000)
    
    def step(self):
        """Step update"""
        # Locate the household agent
        home_agents = self.model.grid.get_cell_list_contents([self.home_id])
        if home_agents:
            home_agent = home_agents[0]
            remittance = self.calculate_remittance()
            home_agent.total_remittance += remittance

class MigrationModel(Model):
    """Main model"""
    def __init__(self, num_households: int = 50, num_migrants: int = 10, 
                 shock_probability: float = 0.1):
        super().__init__()
        self.num_households = num_households
        self.num_migrants = num_migrants
        self.shock_probability = shock_probability
        self.schedule = RandomActivation(self)
        
        # Initialize the network
        self.G = nx.Graph()
        self.G.add_nodes_from(range(num_households))
        self.grid = NetworkGrid(self.G)
        
        # Create households
        self.households = []
        self.create_households()
        
        # Create migrants
        self.create_migrants()
        
        # Build the network
        self._build_networks()
        
        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Average Child Marriage Risk": lambda m: np.mean([h.child_marriage_risk for h in m.households]),
                "Total Remittances": lambda m: sum(h.total_remittance for h in m.households),
                "International Migration Ratio": lambda m: self.calculate_international_ratio()
            },
            agent_reporters={
                "Child Marriage Risk": lambda a: getattr(a, "child_marriage_risk", None),
                "Total Remittance": lambda a: getattr(a, "total_remittance", None)
            }
        )
    
    def create_households(self):
        """Create household agents"""
        # Ethnic distribution based on CVFS data
        ethnicity_weights = {
            "Bramin": 0.42, "Hill": 0.19, "Dalit": 0.12,
            "Newar": 0.06, "Terai": 0.18, "Others": 0.03
        }
        
        for i in range(self.num_households):
            ethnicity = random.choices(list(ethnicity_weights.keys()),
                                    weights=list(ethnicity_weights.values()))[0]
            location = (random.uniform(0, 100), random.uniform(0, 100))
            household = HouseholdAgent(i, self, ethnicity, location)
            self.schedule.add(household)
            self.grid.place_agent(household, i)
            self.households.append(household)
    
    def create_migrants(self):
        """Create migrant agents"""
        for i in range(self.num_migrants):
            home_id = random.randint(0, self.num_households - 1)
            # 40% probability of international migration
            if random.random() < 0.4:
                location = f"F{random.randint(1, 974):03d}"
            else:
                location = f"N{random.randint(1, 99):02d}"
            
            migrant = MigrantAgent(self.num_households + i, self, home_id, location)
            self.schedule.add(migrant)
            self.households[home_id].migrants.append(migrant.unique_id)
    
    def _build_networks(self):
        """Build the social network"""
        connection_probability = 0.3
        distance_threshold = 15
        
        # Geographic network
        for h1 in self.households:
            for h2 in self.households:
                if h1 != h2:
                    distance = np.sqrt(sum((a - b) ** 2 for a, b in 
                                        zip(h1.location, h2.location)))
                    if (distance < distance_threshold and 
                        random.random() < connection_probability):
                        self.G.add_edge(h1.unique_id, h2.unique_id)
        
        # Ethnic network
        for h1 in self.households:
            same_ethnic = [h2 for h2 in self.households 
                         if h2.ethnicity == h1.ethnicity and h2 != h1]
            for h2 in same_ethnic:
                if random.random() < connection_probability:
                    self.G.add_edge(h1.unique_id, h2.unique_id)
    
    def calculate_international_ratio(self) -> float:
        """Calculate the ratio of international migrants"""
        international = sum(1 for a in self.schedule.agents 
                          if isinstance(a, MigrantAgent) and a.is_international)
        return international / self.num_migrants if self.num_migrants > 0 else 0
    
    def step(self):
        """Model step"""
        self.datacollector.collect(self)
        self.schedule.step()
        
        # Economic shock
        if random.random() < self.shock_probability:
            shock_magnitude = random.uniform(0.1, 0.3)
            for household in self.households:
                household.income *= (1 - shock_magnitude)

def network_portrayal(G: nx.Graph) -> Dict[str, list]:
    """Network visualization settings"""
    portrayal = {"nodes": [], "edges": []}
    
    # Add nodes
    for node in G.nodes():
        agents = G.nodes[node].get("agent", [])
        if not agents:
            continue
        
        agent = agents[0]
        if isinstance(agent, HouseholdAgent):
            # Node size based on the logarithm of total remittance
            size = 6 + np.log1p(agent.total_remittance / 10000)
            
            portrayal["nodes"].append({
                "id": node,
                "size": size,
                "color": ETHNICITY_PARAMS[agent.ethnicity]["color"],
                "label": f"{agent.ethnicity[:3]}",
                "borderWidth": 2,
                "borderColor": "#FFFFFF"
            })
    
    # Add edges
    for edge in G.edges():
        portrayal["edges"].append({
            "source": edge[0],
            "target": edge[1],
            "color": "#CCCCCC",
            "width": 0.5,
            "alpha": 0.3
        })
    
    return portrayal

# Create visualization components
grid = NetworkModule(network_portrayal, 600, 600)
charts = ChartModule([
    {"Label": "Average Child Marriage Risk", "Color": "#E41A1C"},
    {"Label": "Total Remittances", "Color": "#377EB8"},
    {"Label": "International Migration Ratio", "Color": "#4DAF4A"}
])

# Set model parameters
model_params = {
    "num_households": Slider(
        "Number of Households",
        value=50,
        min_value=10,
        max_value=200,
        step=10
    ),
    "num_migrants": Slider(
        "Number of Migrants",
        value=10,
        min_value=0,
        max_value=50,
        step=5
    ),
    "shock_probability": Slider(
        "Economic Shock Probability",
        value=0.1,
        min_value=0,
        max_value=0.5,
        step=0.05
    )
}

# Create server
server = ModularServer(
    MigrationModel,
    [grid, charts],
    "CVFS Migration and Child Marriage Model",
    model_params
)

# Launch server
if __name__ == '__main__':
    server.launch()
