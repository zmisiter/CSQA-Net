import numpy as np
import itertools
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

np.seterr(divide='ignore',invalid='ignore')


class PartsResort():  
    def __init__(self, num_center, feature_dim):
        super().__init__()
        self.num_center = num_center
        self.feature_dim = feature_dim

        self.centers = np.zeros([num_center, feature_dim])
        self.count = 0

        self.permutations = list(itertools.permutations(range(num_center))) 

    def update(self, points, order):
        batch = points.shape[0] 

        # [batch, topN, feature_dim]
        resorted_points = np.zeros_like(points)
        for i in range(batch):
            resorted_points[i] = points[i][order[i], :]  # shape: [batch, topN]

        # [topN, feature_dim]
        resorted_points = np.mean(resorted_points, axis=0)
        for i in range(self.num_center):
            self.centers[i] = (self.centers[i] * self.count * 0.9 + resorted_points[i] * batch) / (
                        self.count * 0.9 + batch)
        self.count += batch

    def classify(self, points, is_train):
        # input: points [batch, topN, feature_dim]
        # output: [batch, topN]
        batch, topN, _ = points.shape
        if np.sum(self.count) == 0: 
            order = np.stack([list(range(topN))] * batch, axis=0)
            # self.update(points, order)
        else:
            order = np.zeros([batch, topN], dtype=np.int)
            for i in range(points.shape[0]):
                topn_points = points[i]
                order[i] = self.graph_assign(topn_points)
        if is_train:
            self.update(points, order)
        return order

    def graph_assign(self, topn_points): 
        adj_matrix_center = np.dot(self.centers, self.centers.transpose())  
        adj_matrix = np.dot(topn_points, topn_points.transpose())  
        adj_matrix_center = adj_matrix_center / adj_matrix_center.max()
        adj_matrix = adj_matrix / adj_matrix.max() 

        max_similarity = 0
        order = list(range(self.num_center))
        for perm in self.permutations:  
            adj_matrix = adj_matrix[:, perm][perm, :]
            prod = np.sum(adj_matrix_center * adj_matrix)
            if prod > max_similarity: 
                max_similarity = prod
                order = list(perm)
            # print(max_similarity, prod, order)
        return order

def list_loss(logits, targets):
    # temp = F.log_softmax(logits, -1) 
    # loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    # return torch.stack(loss)
    p = nn.CrossEntropyLoss(reduction='none')
    loss = p(logits, targets)
    return loss  

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def ranking_loss(score, targets):   # targets:[B, 4]
    if torch.cuda.is_available():
        loss = torch.zeros(1).cuda()   
        data_type = torch.cuda.FloatTensor
    else:
        loss = torch.zeros(1)
        data_type = torch.FloatTensor
    batch_size = score.size(0) 

    for i in range(targets.shape[1]): 
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(data_type)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size


def smooth_CE(logits, label, peak):
    # logits - [batch, num_cls]
    # label - [batch]
    batch, num_cls = logits.shape
    label_logits = np.zeros(logits.shape, dtype=np.float32) + (1-peak)/(num_cls-1)
    ind = ([i for i in range(batch)], list(label.data.cpu().numpy()))
    label_logits[ind] = peak
    smooth_label = torch.from_numpy(label_logits).to(logits.device)

    logits = F.log_softmax(logits, -1)
    ce = torch.mul(logits, smooth_label)
    loss = torch.mean(-torch.sum(ce, -1)) # batch average

    return loss


def keep_top_k_row(matrix, k):
    batch_size, num_nodes, _, _= matrix.shape
    values, indices = torch.topk(matrix, k, dim=-1)
    result = torch.zeros_like(matrix)
    
    # values = torch.gather(matrix,-1,indices)
    result.scatter_(-1,indices,values)

    return result

if __name__ == "__main__":
    points = np.random.randint(0, 10, size=[2, 6, 105])
