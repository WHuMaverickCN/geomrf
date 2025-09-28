import torch

import torch.nn as nn
import torch.optim as optim

import numpy as np

class GraphicalMarkovRandomField:
    def __init__(self, num_nodes, edges, beta=0.5, lambda_param=0.1):
        self.num_nodes = num_nodes  # 节点总数
        self.edges = edges  # 边的列表，每个元素是一对节点 (i, j)
        self.beta = beta  # β系数，控制平滑项的强度
        self.lambda_param = lambda_param  # λ系数，控制数据项和平滑项的权衡
        self.labels = np.zeros(num_nodes)  # 初始化节点标签

    def P_both(self, G):
        # 这里需要定义返回每个节点的概率的逻辑
        return np.random.rand(self.num_nodes)  # 随机概率，应根据实际模型调整

    def energy_data(self, L_G, idx):
        # 计算数据能量
        if L_G == 0:
            return self.P_both(idx)
        else:
            return 1 - self.P_both(idx)

    def energy_smooth(self, L_Gi, L_Gj):
        # 计算平滑能量
        if L_Gi == L_Gj:
            return 0
        else:
            return 1 - self.P_both(G)

    def total_energy(self):
        # 计算总能量
        total = 0
        p_both = self.P_both(None)  # 假设 P_both 对所有节点计算一次足够
        for i in range(self.num_nodes):
            total += self.lambda_param * self.energy_data(self.labels[i], i)
            for j in self.edges[i]:
                total += self.beta * self.energy_smooth(self.labels[i], self.labels[j])
        return total

    def simulate(self):
        # 模拟 MRF 过程
        for i in range(self.num_nodes):
            # 模拟更新节点 i 的标签
            self.labels[i] = np.random.choice([0, 1], p=[0.5, 0.5])  # 随机选择，应根据实际概率调整



class CRF(nn.Module):
    def __init__(self, tagset_size):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size

        # Transition matrix for CRF
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)
        )

        # Initialize transitions to not transition to start and from end
        self.transitions.data[:, 0] = -10000
        self.transitions.data[1, :] = -10000

    def forward(self, feats):
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][0] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[1]
        best_tag_id = torch.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == 0
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, feats, tags):
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][0] = 0

        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[1]
        alpha = torch.logsumexp(terminal_var, dim=1)[0]
        return alpha

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([0], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[1, tags[-1]]
        return score

# Example usage
tagset_size = 5  # Example tagset size
model = CRF(tagset_size)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Example input
feats = torch.randn(10, tagset_size)  # Example feature tensor
tags = torch.tensor([1, 2, 3, 4, 0, 1, 2, 3, 4, 0], dtype=torch.long)  # Example tag sequence

# Training step
model.zero_grad()
loss = model.neg_log_likelihood(feats, tags)
loss.backward()
optimizer.step()

# Decoding step
with torch.no_grad():
    score, tag_seq = model(feats)
    print("Score:", score)
    print("Tag sequence:", tag_seq)