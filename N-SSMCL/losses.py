#losses.py
import torch
import torch.nn.functional as F

def contrastive_loss(features_a, features_b, temperature=1.0):
    batch_size = features_a.size(0)
    features = torch.cat([features_a, features_b], dim=0)
    sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
    sim_matrix = sim_matrix / temperature
    labels = torch.arange(batch_size, device=features_a.device)
    labels = torch.cat([labels, labels], dim=0)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def contrastive_loss_ce_hard_negatives(features_a, features_b, temperature=0.1, num_negatives=2):
    batch_size = features_a.size(0)
    features_a = F.normalize(features_a, dim=1)
    features_b = F.normalize(features_b, dim=1)
    all_features = torch.cat([features_a, features_b], dim=0)
    sim_matrix = torch.matmul(features_a, all_features.T) / temperature
    pos_indices = torch.arange(batch_size, device=features_a.device)
    sim_matrix[torch.arange(batch_size), pos_indices + batch_size] = float('-inf')
    _, neg_indices = torch.topk(sim_matrix, k=num_negatives, dim=1, largest=False)
    pos_sim = torch.sum(features_a * features_b, dim=1, keepdim=True) / temperature
    neg_sims = torch.gather(sim_matrix, 1, neg_indices)
    logits = torch.cat([pos_sim, neg_sims], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=features_a.device)
    loss = F.cross_entropy(logits, labels)
    return loss
