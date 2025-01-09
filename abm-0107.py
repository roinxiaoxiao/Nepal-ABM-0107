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

# 族群参数设置
ETHNICITY_PARAMS = {
    "Bramin": {"gender_norm_base": 0.6, "income_base": 1500, "education_tendency": 0.3},
    "Hill": {"gender_norm_base": 0.5, "income_base": 1200, "education_tendency": 0.25},
    "Dalit": {"gender_norm_base": 0.4, "income_base": 800, "education_tendency": 0.2},
    "Newar": {"gender_norm_base": 0.7, "income_base": 1800, "education_tendency": 0.35},
    "Terai": {"gender_norm_base": 0.5, "income_base": 1000, "education_tendency": 0.25},
    "Others": {"gender_norm_base": 0.5, "income_base": 1200, "education_tendency": 0.25}
}

class HouseholdAgent(Agent):
    def __init__(self, unique_id, model, ethnicity, location):
        super().__init__(unique_id, model)
        self.ethnicity = ethnicity
        self.location = location
        self.pos = unique_id  # 添加位置属性

        # 基于族群特征初始化参数
        eth_params = ETHNICITY_PARAMS[ethnicity]
        base_income = eth_params["income_base"]
        self.income = random.uniform(base_income * 0.8, base_income * 1.2)
        self.gender_norm = random.uniform(
            eth_params["gender_norm_base"] - 0.1,
            eth_params["gender_norm_base"] + 0.1
        )
        self.education_tendency = eth_params["education_tendency"]
        
        # 其他属性初始化
        self.remittance = 0
        self.education_investment = 0
        self.child_marriage_risk = random.uniform(0.4, 0.8)
        self.migrants = []
        self.shock_resistance = random.uniform(0.3, 0.7)
        
    def step(self):
        # 更新收入（包括汇款）
        total_income = self.income + self.remittance
        
        # 计算教育投资
        self.education_investment = (total_income * self.education_tendency)
        
        # 更新童婚风险
        if self.gender_norm > 0.5:
            self.child_marriage_risk -= 0.05
            
        education_effect = (self.education_investment / total_income) * 0.1
        self.child_marriage_risk -= education_effect
        
        economic_effect = (total_income / 2000) * 0.05
        self.child_marriage_risk -= economic_effect
        
        # 更新来自邻居的影响
        self.update_from_neighbors()
        
        # 确保风险在合理范围内
        self.child_marriage_risk = max(0.1, min(0.9, self.child_marriage_risk))
        
    def update_from_neighbors(self):
        """更新来自邻居的影响"""
        neighbor_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbors = [self.model.schedule.agents[n] for n in neighbor_nodes 
                    if isinstance(self.model.schedule.agents[n], HouseholdAgent)]
        
        if not neighbors:
            return
            
        # 计算邻居影响
        same_ethnicity_weight = 0.7
        diff_ethnicity_weight = 0.3
        
        weighted_norm = 0
        total_weight = 0
        
        for neighbor in neighbors:
            weight = same_ethnicity_weight if neighbor.ethnicity == self.ethnicity else diff_ethnicity_weight
            weighted_norm += neighbor.gender_norm * weight
            total_weight += weight
            
        if total_weight > 0:
            avg_norm = weighted_norm / total_weight
            self.gender_norm = 0.8 * self.gender_norm + 0.2 * avg_norm
            
        # 更新童婚风险
        avg_risk = np.mean([n.child_marriage_risk for n in neighbors])
        self.child_marriage_risk = 0.8 * self.child_marriage_risk + 0.2 * avg_risk

class MigrantAgent(Agent):
    def __init__(self, unique_id, model, home_id, current_location):
        super().__init__(unique_id, model)
        self.home_id = home_id
        self.current_location = current_location
        self.gender_norm_influence = random.uniform(0.1, 0.3)
        self.remittance_ability = random.uniform(100, 500)
        self.success_rate = random.uniform(0.6, 1.0)
        
    def step(self):
        if self.home_id in self.model.schedule.agents:
            home_agent = self.model.schedule.agents[self.home_id]
            
            # 传播性别平等观念
            norm_change = self.gender_norm_influence * self.success_rate
            home_agent.gender_norm = min(1, home_agent.gender_norm + norm_change)
            
            # 进行汇款
            remittance = self.remittance_ability * self.success_rate * random.uniform(0.8, 1.2)
            home_agent.remittance = remittance

class MigrationModel(Model):
    def __init__(self, num_households, num_migrants, shock_probability=0.1):
        super().__init__()
        self.num_households = num_households
        self.num_migrants = num_migrants
        self.shock_probability = shock_probability
        self.schedule = RandomActivation(self)
        
        # 创建网络
        G = nx.Graph()
        G.add_nodes_from(range(num_households))
        self.grid = NetworkGrid(G)
        
        self.households = []
        
        # 创建家庭代理人
        for i in range(self.num_households):
            ethnicity = random.choice(list(ETHNICITY_PARAMS.keys()))
            location = (random.uniform(0, 100), random.uniform(0, 100))
            household = HouseholdAgent(i, self, ethnicity, location)
            self.schedule.add(household)
            self.grid.place_agent(household, i)
            self.households.append(household)
            
        # 创建迁移代理人
        for i in range(self.num_migrants):
            home_id = random.randint(0, self.num_households - 1)
            current_location = (random.uniform(100, 200), random.uniform(100, 200))
            migrant = MigrantAgent(self.num_households + i, self, home_id, current_location)
            self.schedule.add(migrant)
            
            # 将迁移者添加到家庭
            self.households[home_id].migrants.append(migrant.unique_id)
            
        # 构建网络连接
        self._build_networks()
        
        # 设置数据收集器
        self.datacollector = DataCollector(
            model_reporters={
                "Average Child Marriage Risk": lambda m: self.calculate_avg_metric("child_marriage_risk"),
                "Average Gender Norm": lambda m: self.calculate_avg_metric("gender_norm"),
                "Total Remittances": lambda m: self.calculate_total_remittances(),
            },
            agent_reporters={
                "Child Marriage Risk": lambda a: getattr(a, "child_marriage_risk", None),
                "Gender Norm": lambda a: getattr(a, "gender_norm", None),
                "Ethnicity": lambda a: getattr(a, "ethnicity", None) if isinstance(a, HouseholdAgent) else None
            }
        )
        
    def _build_networks(self):
        """构建社交网络"""
        # 基于地理位置的连接
        for household in self.households:
            for other_household in self.households:
                if household != other_household:
                    distance = ((household.location[0] - other_household.location[0]) ** 2 +
                              (household.location[1] - other_household.location[1]) ** 2) ** 0.5
                    if distance < 20:  # 距离阈值
                        self.grid.graph.add_edge(household.unique_id, other_household.unique_id)
        
        # 基于族群的连接
        for household in self.households:
            for other_household in self.households:
                if (household != other_household and 
                    household.ethnicity == other_household.ethnicity):
                    self.grid.graph.add_edge(household.unique_id, other_household.unique_id)
                    
    def calculate_avg_metric(self, metric):
        """计算家庭代理人的平均指标"""
        values = [getattr(agent, metric) for agent in self.households]
        return np.mean(values) if values else 0
        
    def calculate_total_remittances(self):
        """计算总汇款金额"""
        return sum(household.remittance for household in self.households)
        
    def step(self):
        # 收集数据
        self.datacollector.collect(self)
        
        # 执行步进
        self.schedule.step()
        
        # 随机经济冲击
        if random.random() < self.shock_probability:
            shock_magnitude = random.uniform(0.1, 0.3)
            for household in self.households:
                impact = shock_magnitude * (1 - household.shock_resistance)
                household.income = max(household.income * (1 - impact), household.income * 0.6)

def network_portrayal(G):
    portrayal = {"nodes": [], "edges": []}
    
    for node in G.nodes():
        agent = G.nodes[node].get("agent", [])[0] if G.nodes[node].get("agent", []) else None
        if agent and isinstance(agent, HouseholdAgent):
            color = {
                "Bramin": "#1f77b4",
                "Hill": "#ff7f0e",
                "Dalit": "#2ca02c",
                "Newar": "#d62728",
                "Terai": "#9467bd",
                "Others": "#8c564b"
            }[agent.ethnicity]
            
            portrayal["nodes"].append({
                "id": node,
                "size": 5 + (agent.income / 500),
                "color": color,
                "label": f"{agent.ethnicity[:3]}"
            })
    
    for edge in G.edges():
        portrayal["edges"].append({
            "source": edge[0],
            "target": edge[1],
            "color": "#000000",
            "width": 1
        })
    
    return portrayal

# 创建可视化组件
network = NetworkModule(network_portrayal, 500, 500)
charts = ChartModule([
    {"Label": "Average Child Marriage Risk", "Color": "Red"},
    {"Label": "Average Gender Norm", "Color": "Blue"},
    {"Label": "Total Remittances", "Color": "Green"}
])

# 设置模型参数
model_params = {
    "num_households": Slider(
        "Number of Households",
        value=50,
        min_value=10,
        max_value=100,
        step=5
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

# 创建服务器
server = ModularServer(
    MigrationModel,
    [network, charts],
    "Migration and Child Marriage Model",
    model_params
)

if __name__ == '__main__':
    server.launch()