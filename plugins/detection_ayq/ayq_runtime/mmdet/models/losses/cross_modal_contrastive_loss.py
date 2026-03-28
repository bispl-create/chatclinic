# 경로: mmdet/models/losses/cross_modal_contrastive_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

@MODELS.register_module()
class CrossModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, query_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        bs, nq, dim = query_feats.shape
        _, feat_num, _ = text_feats.shape

        query_feats = F.normalize(query_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)
        text_feats = text_feats.squeeze(1)
        # text_feats = text_feats.mean(dim=1)  # [bs, dim]

        query_feats = query_feats.view(bs * nq, dim)
        logits = torch.matmul(query_feats, text_feats.T) / self.temperature
            
        targets = torch.arange(bs).repeat_interleave(nq).to(query_feats.device)

        # targets: [0,0,0…1,1,1…] 형태로 이미 정의된 상태
        pos = logits[torch.arange(bs*nq), targets]       # positive column
        neg = logits[torch.arange(bs*nq), 1 - targets]   # negative column
        # print("pos mean", pos.mean().item(), "neg mean", neg.mean().item())

        loss = F.cross_entropy(logits, targets, reduction=self.reduction)

        # modified for saving npy features
        return loss, query_feats
