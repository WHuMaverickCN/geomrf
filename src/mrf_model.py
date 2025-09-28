import numpy as np
import yaml
# PLAUSIBLE = 0.9
# SUSPICIOUS= 0.1
# LAMBDA_PARAM=0.1
# BETA = 0.5

with open('./config_sf.yaml', 'r') as f:
    sf_config = yaml.safe_load(f)

params = sf_config.get('params', {})
WINDOW_SIZE = params.get('window_size', 100)
PLAUSIBLE = params.get('plausible', 0.95)
SUSPICIOUS = params.get('suspicious', 0.05)
LAMBDA_PARAM = params.get('lambda_param', 0.1)
BETA = params.get('beta', 0.5)

class GraphicalMarkovRandomField:
    def __init__(self, nodes, edges,node_eval, beta=BETA, lambda_param=LAMBDA_PARAM):
        self.nodes = nodes  # 节点的字典，键是节点ID，值可以是标签或其他属性
        self.edges = edges  # 边的字典，键是节点ID，值是邻接节点ID的列表
        self.beta = beta  # β系数，控制平滑项的强度
        self.lambda_param = lambda_param  # λ系数，控制数据项和平滑项的权衡
        self.labels = {node: 0 for node in nodes}  # 初始化节点标签为0
        self.node_eval = node_eval

    def P_both(self, node_id):
        # 这里返回每个节点的概率，假设为随机概率，应根据实际模型调整
        if self.node_eval[node_id]=='准确' or self.node_eval[node_id]==1:
            return PLAUSIBLE
        elif self.node_eval[node_id]=='不准确' or self.node_eval[node_id]==0:
            return SUSPICIOUS
        # return np.random.rand()

    def P_both_smooth(self, node_id,neighbor_id):
        return abs(self.P_both(node_id)-self.P_both(neighbor_id))
    def energy_data(self, L_G, node_id):
        # 计算数据能量
        if L_G == 0:
            return self.P_both(node_id)
        else:
            return 1 - self.P_both(node_id)

    def energy_smooth(self, L_Gi, L_Gj,id_Gi,id_Gj):
        # 计算平滑能量
        if L_Gi == L_Gj:
            return 0
        else:
            return 1 - self.P_both_smooth(id_Gi,id_Gj)  # 也可以根据节点ID计算不同的平滑能量

    def total_energy_mine(self):
        # 计算总能量
        total = 0
        for node_id in self.nodes:
            total += self.lambda_param * self.energy_data(self.labels[node_id], node_id)
            for neighbor_id in self.edges[node_id]:
                total += self.beta * self.energy_smooth(self.labels[node_id], self.labels[neighbor_id],node_id,neighbor_id)
        return total
    # def total_energy(self):
    #     # 计算总能量
    #     total = 0
    #     for node_id in self.nodes:
    #         total += self.lambda_param * self.energy_data(self.labels[node_id], node_id)
    #         for neighbor_id in self.edges[node_id]:
    #             total += self.beta * self.energy_smooth(self.labels[node_id], self.labels[neighbor_id])
    #     return total

    def simulate(self):
        # 模拟 MRF 过程
        for node_id in self.nodes:
            # 模拟更新节点 node_id 的标签
            # self.labels[node_id] = np.random.choice([0, 1], p=[0.1, 0.9])  # 随机选择标签
            self.labels[node_id] = np.random.choice([0, 1], p=[SUSPICIOUS, PLAUSIBLE])  # 随机选择标签

