# mmdet/models/text_encoders/on_the_fly_text_encoder.py
from __future__ import annotations
import functools
import torch
import torch.nn as nn
from mmdet.registry import MODELS

@MODELS.register_module()
class OnTheFlyTextEncoder(nn.Module):
    """런타임에 텍스트 -> 임베딩을 생성하는 래퍼.
    model_type: 'biomedclip' | 'openclip' | 'pubmedclip_hf'
    """
    def __init__(self, model_type='biomedclip', device=None, freeze=True):
        super().__init__()
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._build()

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        # 같은 문자열은 재사용 (CPU 캐시)
        self._encode_cached = functools.lru_cache(maxsize=100000)(self._encode_single_str)

    def _build(self):
        if self.model_type == 'biomedclip':
            import open_clip
            name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            self.model, _, _ = open_clip.create_model_and_transforms(name)
            self.model.to(self.device).eval()
            self.tokenizer = open_clip.get_tokenizer(name)
            self.context_length = 256
            self._encode_batch = self._encode_batch_openclip

        elif self.model_type == 'openclip':
            import open_clip
            name = 'ViT-B-32-quickgelu'
            self.model, _, _ = open_clip.create_model_and_transforms(name, pretrained='openai')
            self.model.to(self.device).eval()
            self.tokenizer = open_clip.get_tokenizer(name)
            self.context_length = 77
            self._encode_batch = self._encode_batch_openclip

        elif self.model_type == 'pubmedclip_hf':
            from transformers import AutoTokenizer, CLIPModel
            ckpt = 'flaviagiammarino/pubmed-clip-vit-base-patch32'
            self.model = CLIPModel.from_pretrained(ckpt).to(self.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
            self._encode_batch = self._encode_batch_hf

        else:
            raise ValueError(f'Unknown model_type: {self.model_type}')

    @torch.no_grad()
    def _encode_batch_openclip(self, texts):
        tokens = self.tokenizer(texts, context_length=self.context_length).to(self.device)
        feats = self.model.encode_text(tokens)  # [B, Dt]
        return feats

    @torch.no_grad()
    def _encode_batch_hf(self, texts):
        toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        feats = self.model.get_text_features(**toks)  # [B, Dt]
        return feats

    def _encode_single_str(self, text: str):
        feats = self._encode_batch([text]).squeeze(0).detach().cpu()  # Tensor[Dt] on CPU
        return feats

    @torch.no_grad()
    def encode_texts(self, texts):
        """list[str] -> Tensor[B, Dt] (GPU)"""
        if len(texts) <= 8:  # 소량이면 캐시 이득 큼
            out = [self._encode_cached(t) for t in texts]  # CPU Tensor들
            return torch.stack(out, dim=0).to(self.device)
        return self._encode_batch(texts).to(self.device)
