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

# Ethnic group parameters
ETHNICITY_PARAMS = {
    "Bramin": {
        "income_base": 1500, 
        "migration_tendency": 0.3, 
        "color": "#E41A1C",
        "education_weight": 0.3
    },
    "Hill": {
        "income_base": 1200, 
        "migration_tendency": 0.25, 
        "color": "#377EB8",
        "education_weight": 0.25
    },
    "Dalit": {
        "income_base": 800, 
        "migration_tendency": 0.2, 
        "color": "#4DAF4A",
        "education_weight": 0.2
    },
    "Newar": {
        "income_base": 1800, 
        "migration_tendency": 0.35, 
        "color": "#984EA3",
        "education_weight": 0.35
    },
    "Terai": {
        "income_base": 1000, 
        "migration_tendency": 0.25, 
        "color": "#FF7F00",
        "education_weight": 0.25
    },
    "Others": {
        "income_base": 1200, 
        "migration_tendency": 0.25, 
        "color": "#A65628",
        "education_weight": 0.25
    }
}

class HouseholdAgent(Agent):
    """Household agent"""
    def __init__(self, unique_id: int, model: Model, ethnicity: str, location: tuple):
        super().__init__(unique_id, model)
        self.ethnicity = ethnicity
        self.location = location
        self.pos = unique_id
        
        # Base attributes
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
        
        # Spillover-related attributes
        self.received_spillover_effect = 0
        self.neighbor_migrant_ratio = 0
        self.knowledge_spillover = 0
        self.economic_spillover = 0
        self.norm_spillover = 0
    
    def calculate_spillover_effects(self):
        """Calculate spillover effects"""
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        if not neighbors:
            return
            
        # Retrieve neighbor agents
        neighbor_agents = []
        for n in neighbors:
            if n < len(self.model.schedule.agents):
                agent = self.model.schedule.agents[n]
                if isinstance(agent, HouseholdAgent):
                    neighbor_agents.append(agent)
        
        if not neighbor_agents:
            return
            
        # Calculate the proportion of neighbors with migrants
        migrant_neighbors = [n for n in neighbor_agents if n.migrants]
        self.neighbor_migrant_ratio = len(migrant_neighbors) / len(neighbor_agents)
        
        # Calculate knowledge spillover
        total_knowledge = 0
        weights = 0
        for neighbor in neighbor_agents:
            # Distance weight
            distance = np.sqrt(sum((a - b) ** 2 for a, b in 
                             zip(self.location, neighbor.location)))
            distance_weight = 1 / (1 + distance)
            
            # Ethnic weight
            ethnic_weight = 1.5 if neighbor.ethnicity == self.ethnicity else 1.0
            
            # Migrant status weight
            migrant_weight = 1.5 if neighbor.migrants else 1.0
            
            total_weight = distance_weight * ethnic_weight * migrant_weight
            total_knowledge += neighbor.education_investment * total_weight
            weights += total_weight
        
        self.knowledge_spillover = total_knowledge / weights if weights > 0 else 0
        
        # Calculate economic spillover
        self.economic_spillover = sum([n.total_remittance * 0.05 
                                     for n in migrant_neighbors])
        
        # Calculate norm spillover
        norm_total = 0
        norm_weights = 0
        for neighbor in neighbor_agents:
            eth_weight = 1.5 if neighbor.ethnicity == self.ethnicity else 1.0
            mig_weight = 1.5 if neighbor.migrants else 1.0
            weight = eth_weight * mig_weight
            norm_total += (1 - neighbor.child_marriage_risk) * weight
            norm_weights += weight
        
        self.norm_spillover = norm_total / norm_weights if norm_weights > 0 else 0
        
        # Calculate total spillover effect
        self.received_spillover_effect = (
            self.knowledge_spillover * 0.4 +
            self.economic_spillover * 0.3 +
            self.norm_spillover * 0.3
        )

    def update_development_indicators(self):
        """Update development indicators"""
        if self.total_remittance > 0:
            self.education_investment = min(self.total_remittance * 0.3, 100000)
        
        # Update child marriage risk
        remittance_effect = -0.05 * (self.total_remittance / 50000)
        education_effect = -0.1 * (self.education_investment / 10000)
        spillover_effect = -0.1 * self.received_spillover_effect
        
        self.child_marriage_risk = max(0.1, min(0.9,
            self.child_marriage_risk + remittance_effect + 
            education_effect + spillover_effect
        ))
    
    def step(self):
        """Step update"""
        self.calculate_spillover_effects()
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
        if self.home_id < len(self.model.schedule.agents):
            home_agent = self.model.schedule.agents[self.home_id]
            if isinstance(home_agent, HouseholdAgent):
                remittance = self.calculate_remittance()
                home_agent.total_remittance += remittance

class MigrationModel(Model):
    """Model to simulate migration and spillover effects"""
    def __init__(self, num_households: int = 50, num_migrants: int = 10, 
                 shock_probability: float = 0.1):
        super().__init__()
        self.num_households = num_households
        self.num_migrants = num_migrants
        self.shock_probability = shock_probability
        self.schedule = RandomActivation(self)
        
        # Create network
        self.G = nx.Graph()
        self.G.add_nodes_from(range(num_households))
        self.grid = NetworkGrid(self.G)
        
        # Create households
        self.households = []
        self.create_households()
        
        # Create migrants
        self.create_migrants()
        
        # Build networks
        self._build_networks()
        
        # Set up data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Average Child Marriage Risk": lambda m: np.mean([h.child_marriage_risk for h in m.households]),
                "Total Remittances": lambda m: sum(h.total_remittance for h in m.households),
                "Average Spillover Effect": lambda m: np.mean([h.received_spillover_effect for h in m.households]),
                "Knowledge Spillover": lambda m: np.mean([h.knowledge_spillover for h in m.households]),
                "Economic Spillover": lambda m: np.mean([h.economic_spillover for h in m.households]),
                "Norm Spillover": lambda m: np.mean([h.norm_spillover for h in m.households])
            },
            agent_reporters={
                "Child Marriage Risk": lambda a: getattr(a, "child_marriage_risk", None),
                "Total Remittance": lambda a: getattr(a, "total_remittance", None),
                "Spillover Effect": lambda a: getattr(a, "received_spillover_effect", None)
            }
        )
    def create_households(self):
        """Create household agents"""
        # Ethnic group distribution based on CVFS data
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
            # 40% chance of international migration
            if random.random() < 0.4:
                location = f"F{random.randint(1, 974):03d}"
            else:
                location = f"N{random.randint(1, 99):02d}"
            
            migrant = MigrantAgent(self.num_households + i, self, home_id, location)
            self.schedule.add(migrant)
            self.households[home_id].migrants.append(migrant.unique_id)
    
    def _build_networks(self):
        """Build social networks"""
        connection_probability = 0.3
        distance_threshold = 15
        
        # Geographic location network
        for h1 in self.households:
            for h2 in self.households:
                if h1 != h2:
                    distance = np.sqrt(sum((a - b) ** 2 for a, b in 
                                        zip(h1.location, h2.location)))
                    if (distance < distance_threshold and 
                        random.random() < connection_probability):
                        self.G.add_edge(h1.unique_id, h2.unique_id)
        
        # Ethnic group network
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
    """Set up network visualization"""
    portrayal = {"nodes": [], "edges": []}
    
    # Add nodes
    for node in G.nodes():
        agents = G.nodes[node].get("agent", [])
        if not agents:
            continue
        
        agent = agents[0]
        if isinstance(agent, HouseholdAgent):
            # Node size based on spillover effect and remittance
            size = 6 + np.log1p(agent.received_spillover_effect + 1) * 3
            
            # Nodes with migrants appear larger
            if agent.migrants:
                size += 2
            
            portrayal["nodes"].append({
                "id": node,
                "size": size,
                "color": ETHNICITY_PARAMS[agent.ethnicity]["color"],
                "label": f"{agent.ethnicity[:3]}\n{agent.received_spillover_effect:.1f}",
                "borderWidth": 2,
                "borderColor": "white" if agent.migrants else "black"
            })
    
    # Add edges, with transparency based on the strength of spillover effects
    for edge in G.edges():
        source = G.nodes[edge[0]]["agent"][0]
        target = G.nodes[edge[1]]["agent"][0]
        
        if isinstance(source, HouseholdAgent) and isinstance(target, HouseholdAgent):
            edge_weight = (source.received_spillover_effect + 
                         target.received_spillover_effect) / 2
            
            # Connections within the same ethnic group are more prominent
            if source.ethnicity == target.ethnicity:
                alpha = min(edge_weight * 0.15 + 0.2, 0.7)
                width = 1.0
            else:
                alpha = min(edge_weight * 0.1, 0.4)
                width = 0.5
            
            portrayal["edges"].append({
                "source": edge[0],
                "target": edge[1],
                "color": f"rgba(0,0,0,{alpha})",
                "width": width + min(edge_weight * 0.1, 1.5)
            })
    
    return portrayal

# Create visualization components
grid = NetworkModule(network_portrayal, 600, 600)

# Create multiple charts for displaying various metrics
charts = [
    ChartModule([
        {"Label": "Average Child Marriage Risk", "Color": "#E41A1C"},
        {"Label": "Average Spillover Effect", "Color": "#377EB8"}
    ]),
    ChartModule([
        {"Label": "Knowledge Spillover", "Color": "#4DAF4A"},
        {"Label": "Economic Spillover", "Color": "#984EA3"},
        {"Label": "Norm Spillover", "Color": "#FF7F00"}
    ])
]

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
    [grid] + charts,  # Combine network graph and all charts
    "CVFS Migration and Child Marriage Model",
    model_params
)

# Launch server
if __name__ == '__main__':
    server.launch()
