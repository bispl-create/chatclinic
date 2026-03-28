# mmdet/models/losses/info_nce_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

@MODELS.register_module()
class InfoNCELoss(nn.Module):
    """InfoNCE / NT-Xent Loss (normalized temperature-scaled cross entropy loss)."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (bs, num_queries, dim)
            labels: (bs,) or (bs * num_queries,)
        Returns:
            loss: scalar
        """
        bs, nq, dim = features.shape
        features = F.normalize(features, dim=-1)
        features = features.view(-1, dim)  # (bs * nq, dim)
        labels = labels.view(-1)

        logits = torch.matmul(features, features.T) / self.temperature  # (N, N)
        logits_mask = ~torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)

        # label similarity mask
        label_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        label_mask = label_mask * logits_mask.float()  # 자기 자신 제외

        # log-softmax
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (label_mask * log_prob).sum(1) / (label_mask.sum(1) + 1e-6)
        loss = -mean_log_prob_pos.mean()

        return loss
