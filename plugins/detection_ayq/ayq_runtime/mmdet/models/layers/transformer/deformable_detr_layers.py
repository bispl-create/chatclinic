# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList
from torch import Tensor, nn

from .detr_layers import (DetrTransformerDecoder, DetrTransformerDecoderLayer,
                          DetrTransformerEncoder, DetrTransformerEncoderLayer)
from .utils import inverse_sigmoid

from typing import Dict, List, Tuple, Union

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


class DeformableDetrTransformerEncoder(DetrTransformerEncoder):
    """Transformer encoder of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor, valid_ratios: Tensor,
                **kwargs) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (bs, num_queries, dim)
        """
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                **kwargs)
        return query

    @staticmethod
    def get_encoder_reference_points(
            spatial_shapes: Tensor, valid_ratios: Tensor,
            device: Union[torch.device, str]) -> Tensor:
        """Get the reference points used in encoder.

        Args:
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            device (obj:`device` or str): The device acquired by the
                `reference_points`.

        Returns:
            Tensor: Reference points used in decoder, has shape (bs, length,
            num_levels, 2).
        """

        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        # [bs, sum(hw), num_level, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points


class DeformableDetrTransformerDecoder(DetrTransformerDecoder):
    """Transformer Decoder of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                reg_branches: Optional[nn.Module] = None,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input queries, has shape (bs, num_queries,
                dim).
            query_pos (Tensor): The input positional query, has shape
                (bs, num_queries, dim). It will be added to `query` before
                forward function.
            value (Tensor): The input values, has shape (bs, num_value, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h) when `as_two_stage` is `True`, otherwise has
                shape (bs, num_queries, 2) with the last dimension arranged
                as (cx, cy).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`, optional): Used for refining
                the regression results. Only would be passed when
                `with_box_refine` is `True`, otherwise would be `None`.

        Returns:
            tuple[Tensor]: Outputs of Deformable Transformer Decoder.

            - output (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * \
                    valid_ratios[:, None]
            output = layer(
                output,
                query_pos=query_pos, # [4, 300, 256]
                value=value, # [4, 16722, 256]
                key_padding_mask=key_padding_mask, # [4, 16722]
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp_reg_preds = reg_branches[layer_id](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp_reg_preds + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp_reg_preds
                    new_reference_points[..., :2] = tmp_reg_preds[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


class DeformableDetrTransformerDecoder_AMODEL(DetrTransformerDecoder):
    """Transformer Decoder of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerDecoderLayer_AMODEL(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                npy_list: List,
                reg_branches: Optional[nn.Module] = None,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input queries, has shape (bs, num_queries,
                dim).
            query_pos (Tensor): The input positional query, has shape
                (bs, num_queries, dim). It will be added to `query` before
                forward function.
            value (Tensor): The input values, has shape (bs, num_value, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h) when `as_two_stage` is `True`, otherwise has
                shape (bs, num_queries, 2) with the last dimension arranged
                as (cx, cy).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`, optional): Used for refining
                the regression results. Only would be passed when
                `with_box_refine` is `True`, otherwise would be `None`.

        Returns:
            tuple[Tensor]: Outputs of Deformable Transformer Decoder.

            - output (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * \
                    valid_ratios[:, None]
            output = layer(
                output,
                npy_list=npy_list,
                query_pos=query_pos, # [4, 300, 256]
                value=value, # [4, 16722, 256]
                key_padding_mask=key_padding_mask, # [4, 16722]
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp_reg_preds = reg_branches[layer_id](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp_reg_preds + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp_reg_preds
                    new_reference_points[..., :2] = tmp_reg_preds[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points

class DeformableDetrTransformerDecoder_ContrastiveMOCA(DetrTransformerDecoder):
    """Transformer Decoder of Deformable DETR."""

    def __init__(self, apply_layers=None, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 원래 부모 init 호출
        self.apply_layers = apply_layers

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerDecoderLayer_ContrastiveMOCA(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                npy_list: List,
                reg_branches: Optional[nn.Module] = None,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input queries, has shape (bs, num_queries,
                dim).
            query_pos (Tensor): The input positional query, has shape
                (bs, num_queries, dim). It will be added to `query` before
                forward function.
            value (Tensor): The input values, has shape (bs, num_value, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h) when `as_two_stage` is `True`, otherwise has
                shape (bs, num_queries, 2) with the last dimension arranged
                as (cx, cy).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`, optional): Used for refining
                the regression results. Only would be passed when
                `with_box_refine` is `True`, otherwise would be `None`.

        Returns:
            tuple[Tensor]: Outputs of Deformable Transformer Decoder.

            - output (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        all_contrastive_features = []

        for layer_id, layer in enumerate(self.layers):
            if self.apply_layers is not None and layer_id > self.apply_layers[-1]:
                continue
            
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * \
                    valid_ratios[:, None]
                
            output = layer(
                output,
                npy_list=npy_list,
                query_pos=query_pos, # [4, 300, 256]
                value=value, # [4, 16722, 256]
                key_padding_mask=key_padding_mask, # [4, 16722]
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                layer_id=layer_id,  # 🔥 이거 추가
                apply_layers=self.apply_layers,  # 🔥 이것도 같이 넘겨
                **kwargs)
            
            
            # if self.training and hasattr(layer, 'saved_contrastive_features'):
            if hasattr(layer, 'saved_contrastive_features'):
                all_contrastive_features.append(layer.saved_contrastive_features[-1])

            if reg_branches is not None:
                tmp_reg_preds = reg_branches[layer_id](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp_reg_preds + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp_reg_preds
                    new_reference_points[..., :2] = tmp_reg_preds[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        
        # if self.training:
        self.saved_contrastive_features = all_contrastive_features

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


class DeformableDetrTransformerDecoder_OnlyContrastive(DetrTransformerDecoder):
    """Transformer Decoder of Deformable DETR."""

    def __init__(self, apply_layers=None, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 원래 부모 init 호출
        self.apply_layers = apply_layers

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerDecoderLayer_OnlyContrastive(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                npy_list: List,
                reg_branches: Optional[nn.Module] = None,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input queries, has shape (bs, num_queries,
                dim).
            query_pos (Tensor): The input positional query, has shape
                (bs, num_queries, dim). It will be added to `query` before
                forward function.
            value (Tensor): The input values, has shape (bs, num_value, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h) when `as_two_stage` is `True`, otherwise has
                shape (bs, num_queries, 2) with the last dimension arranged
                as (cx, cy).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`, optional): Used for refining
                the regression results. Only would be passed when
                `with_box_refine` is `True`, otherwise would be `None`.

        Returns:
            tuple[Tensor]: Outputs of Deformable Transformer Decoder.

            - output (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        all_contrastive_features = []

        for layer_id, layer in enumerate(self.layers):
            if self.apply_layers is not None and layer_id > self.apply_layers[-1]:
                continue
            
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * \
                    valid_ratios[:, None]
                
            output = layer(
                output,
                npy_list=npy_list,
                query_pos=query_pos, # [4, 300, 256]
                value=value, # [4, 16722, 256]
                key_padding_mask=key_padding_mask, # [4, 16722]
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                layer_id=layer_id,  # 🔥 이거 추가
                apply_layers=self.apply_layers,  # 🔥 이것도 같이 넘겨
                **kwargs)
            
            
            # if self.training and hasattr(layer, 'saved_contrastive_features'):
            if hasattr(layer, 'saved_contrastive_features'):
                all_contrastive_features.append(layer.saved_contrastive_features[-1])

            if reg_branches is not None:
                tmp_reg_preds = reg_branches[layer_id](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp_reg_preds + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp_reg_preds
                    new_reference_points[..., :2] = tmp_reg_preds[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        
        # if self.training:
        self.saved_contrastive_features = all_contrastive_features

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points
    

class DeformableDetrTransformerDecoder_OnlyContrastive_DINO(DetrTransformerDecoder):
    """Transformer Decoder of Deformable DETR."""

    def __init__(self, apply_layers=None, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 원래 부모 init 호출
        self.apply_layers = apply_layers

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerDecoderLayer_OnlyContrastive_DINO(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                npy_list: List,
                reg_branches: Optional[nn.Module] = None,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input queries, has shape (bs, num_queries,
                dim).
            query_pos (Tensor): The input positional query, has shape
                (bs, num_queries, dim). It will be added to `query` before
                forward function.
            value (Tensor): The input values, has shape (bs, num_value, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h) when `as_two_stage` is `True`, otherwise has
                shape (bs, num_queries, 2) with the last dimension arranged
                as (cx, cy).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`, optional): Used for refining
                the regression results. Only would be passed when
                `with_box_refine` is `True`, otherwise would be `None`.

        Returns:
            tuple[Tensor]: Outputs of Deformable Transformer Decoder.

            - output (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        all_contrastive_features = []

        for layer_id, layer in enumerate(self.layers):
            if self.apply_layers is not None and layer_id > self.apply_layers[-1]:
                continue
            
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * \
                    valid_ratios[:, None]
                
            output = layer(
                output,
                npy_list=npy_list,
                query_pos=query_pos, # [4, 300, 256]
                value=value, # [4, 16722, 256]
                key_padding_mask=key_padding_mask, # [4, 16722]
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                layer_id=layer_id,  # 🔥 이거 추가
                apply_layers=self.apply_layers,  # 🔥 이것도 같이 넘겨
                **kwargs)
            
            
            # if self.training and hasattr(layer, 'saved_contrastive_features'):
            if hasattr(layer, 'saved_contrastive_features'):
                all_contrastive_features.append(layer.saved_contrastive_features[-1])

            if reg_branches is not None:
                tmp_reg_preds = reg_branches[layer_id](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp_reg_preds + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp_reg_preds
                    new_reference_points[..., :2] = tmp_reg_preds[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        
        # if self.training:
        self.saved_contrastive_features = all_contrastive_features

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points



class DeformableDetrTransformerEncoderLayer(DetrTransformerEncoderLayer):
    """Encoder layer of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)


class DeformableDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Decoder layer of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

class DeformableDetrTransformerDecoderLayer_AMODEL(DetrTransformerDecoderLayer):
        """Decoder layer of Deformable DETR."""

        def _init_layers(self) -> None:
            """Initialize self_attn, cross-attn, ffn, and norms."""
            self.self_attn = MultiheadAttention(**self.self_attn_cfg)
            self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
            self.embed_dims = self.self_attn.embed_dims
            self.ffn = FFN(**self.ffn_cfg)

            self.proj_npy = nn.Linear(512,self.self_attn_cfg['embed_dims'])

            norms_list = [
                build_norm_layer(self.norm_cfg, self.embed_dims)[1]
                for _ in range(3)
            ]
            self.norms = ModuleList(norms_list)
        
        def forward(self,
                query: Tensor,
                npy_list: List,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
            """
            Args:
                query (Tensor): The input query, has shape (bs, num_queries, dim).
                key (Tensor, optional): The input key, has shape (bs, num_keys,
                    dim). If `None`, the `query` will be used. Defaults to `None`.
                value (Tensor, optional): The input value, has the same shape as
                    `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                    `key` will be used. Defaults to `None`.
                query_pos (Tensor, optional): The positional encoding for `query`,
                    has the same shape as `query`. If not `None`, it will be added
                    to `query` before forward function. Defaults to `None`.
                key_pos (Tensor, optional): The positional encoding for `key`, has
                    the same shape as `key`. If not `None`, it will be added to
                    `key` before forward function. If None, and `query_pos` has the
                    same shape as `key`, then `query_pos` will be used for
                    `key_pos`. Defaults to None.
                self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                    (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                    Defaults to None.
                cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                    (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                    Defaults to None.
                key_padding_mask (Tensor, optional): The `key_padding_mask` of
                    `self_attn` input. ByteTensor, has shape (bs, num_value).
                    Defaults to None.

            Returns:
                Tensor: forwarded results, has shape (bs, num_queries, dim).
            """
            '''
            A-Model implementation
            '''

            # 새로운 행과 열을 생성 (1행 1100열, 1100행 1열)
            if self_attn_mask != None:
                new_row = torch.zeros(1, query.shape[1], dtype=torch.bool).to(query.device)  # 새로운 첫 번째 행 (모두 False)
                new_col = torch.zeros(query.shape[1], 1, dtype=torch.bool).to(query.device)  # 새로운 첫 번째 열 (모두 False)
                new_corner = torch.tensor([[False]]).to(query.device)  # (1,1) 크기의 추가된 부분

                # 새로운 Self-Attention Mask 만들기
                extended_row = torch.cat([new_corner, new_row], dim=1)
                extended_self_attn_mask = torch.cat([new_col, self_attn_mask], dim=1)
                self_attn_mask = torch.cat([extended_row, extended_self_attn_mask], dim=0)

            # Query concat
            # npy_list를 PyTorch Tensor로 변환
            prompt_tensor = [torch.tensor(arr).unsqueeze(1) for arr in npy_list]  # 리스트의 NumPy 배열을 Tensor로 변환

            # 텐서를 dim=1 방향으로 concat (batch 차원 유지)
            concatenated_prompt = torch.cat(prompt_tensor, dim=0)
            concatenated_prompt = concatenated_prompt.to(query.device)
            concatenated_prompt = self.proj_npy(concatenated_prompt)

            zero_pos = torch.zeros_like(concatenated_prompt).to(query.device)

            # 원하는 방식으로 Concat 적용
            query = torch.cat([concatenated_prompt, query], dim=1)  # dim=1로 병합
            query_pos = torch.cat([zero_pos, query_pos], dim=1)


            query = self.self_attn(
                query=query,
                key=query,
                value=query,
                query_pos=query_pos,
                key_pos=query_pos,
                attn_mask=self_attn_mask, # None
                **kwargs)
            
            query = query[:,1:,:]
            query_pos = query_pos[:,1:,:]
            
            query = self.norms[0](query)
            query = self.cross_attn(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_mask=cross_attn_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
            query = self.norms[1](query)
            query = self.ffn(query)
            query = self.norms[2](query)

            return query


class DeformableDetrTransformerDecoderLayer_ContrastiveMOCA(DetrTransformerDecoderLayer):
        """Decoder layer of Deformable DETR."""

        ### added code
        #def __init__(self, apply_layers=None, *args, **kwargs):
        #    super().__init__(*args, **kwargs)  # 원래 부모 init 호출
        #    self.apply_layers = apply_layers

        def _init_layers(self) -> None:
            """Initialize self_attn, cross-attn, ffn, and norms."""
            self.self_attn = MultiheadAttention(**self.self_attn_cfg)
            self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
            self.embed_dims = self.self_attn.embed_dims
            self.ffn = FFN(**self.ffn_cfg)

            self.proj_npy = nn.Linear(512,self.self_attn_cfg['embed_dims'])

            norms_list = [
                build_norm_layer(self.norm_cfg, self.embed_dims)[1]
                for _ in range(3)
            ]
            self.norms = ModuleList(norms_list)
        
        def forward(self,
                query: Tensor,
                npy_list: List,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                layer_id: Optional[int] = None,  # 🔥 추가
                apply_layers: Optional[List[int]] = None,  # 🔥 추가
                **kwargs) -> Tensor:
            """
            Args:
                query (Tensor): The input query, has shape (bs, num_queries, dim).
                key (Tensor, optional): The input key, has shape (bs, num_keys,
                    dim). If `None`, the `query` will be used. Defaults to `None`.
                value (Tensor, optional): The input value, has the same shape as
                    `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                    `key` will be used. Defaults to `None`.
                query_pos (Tensor, optional): The positional encoding for `query`,
                    has the same shape as `query`. If not `None`, it will be added
                    to `query` before forward function. Defaults to `None`.
                key_pos (Tensor, optional): The positional encoding for `key`, has
                    the same shape as `key`. If not `None`, it will be added to
                    `key` before forward function. If None, and `query_pos` has the
                    same shape as `key`, then `query_pos` will be used for
                    `key_pos`. Defaults to None.
                self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                    (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                    Defaults to None.
                cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                    (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                    Defaults to None.
                key_padding_mask (Tensor, optional): The `key_padding_mask` of
                    `self_attn` input. ByteTensor, has shape (bs, num_value).
                    Defaults to None.

            Returns:
                Tensor: forwarded results, has shape (bs, num_queries, dim).
            """
            '''
            A-Model implementation
            '''

            # 새로운 행과 열을 생성 (1행 1100열, 1100행 1열)
            if self_attn_mask != None:
                new_row = torch.zeros(1, query.shape[1], dtype=torch.bool).to(query.device)  # 새로운 첫 번째 행 (모두 False)
                new_col = torch.zeros(query.shape[1], 1, dtype=torch.bool).to(query.device)  # 새로운 첫 번째 열 (모두 False)
                new_corner = torch.tensor([[False]]).to(query.device)  # (1,1) 크기의 추가된 부분

                # 새로운 Self-Attention Mask 만들기
                extended_row = torch.cat([new_corner, new_row], dim=1)
                extended_self_attn_mask = torch.cat([new_col, self_attn_mask], dim=1)
                self_attn_mask = torch.cat([extended_row, extended_self_attn_mask], dim=0)

            # Query concat
            # npy_list를 PyTorch Tensor로 변환
            prompt_tensor = [torch.tensor(arr).unsqueeze(1) for arr in npy_list]  # 리스트의 NumPy 배열을 Tensor로 변환

            # 텐서를 dim=1 방향으로 concat (batch 차원 유지)
            concatenated_prompt = torch.cat(prompt_tensor, dim=0)
            concatenated_prompt = concatenated_prompt.to(query.device)
            concatenated_prompt = self.proj_npy(concatenated_prompt)

            zero_pos = torch.zeros_like(concatenated_prompt).to(query.device)

            # 원하는 방식으로 Concat 적용
            query = torch.cat([concatenated_prompt, query], dim=1)  # dim=1로 병합
            query_pos = torch.cat([zero_pos, query_pos], dim=1)


            query = self.self_attn(
                query=query,
                key=query,
                value=query,
                query_pos=query_pos,
                key_pos=query_pos,
                attn_mask=self_attn_mask, # None
                **kwargs)
            
            query = query[:,1:,:]
            query_pos = query_pos[:,1:,:]

            #if self.training:
            self.saved_contrastive_features = []
            # if not hasattr(self, 'saved_conttrastive_features'):
            #     self.saved_contrastive_features = []
            self.saved_contrastive_features.append(query.detach())  # shape: (bs, num_queries, dim)
 
            # 🔥 apply_layers control
            if (apply_layers is not None) and (layer_id == apply_layers[-1]):
                # 만약 apply_layers의 마지막 layer_id와 현재 layer_id가 같다면
                # 여기까지만 계산하고 일찍 return
                del query, query_pos
                return

            query = self.norms[0](query)
            query = self.cross_attn(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_mask=cross_attn_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
            query = self.norms[1](query)
            query = self.ffn(query)
            query = self.norms[2](query)

            return query


class DeformableDetrTransformerDecoderLayer_OnlyContrastive(DetrTransformerDecoderLayer):
        """Decoder layer of Deformable DETR."""

        ### added code
        #def __init__(self, apply_layers=None, *args, **kwargs):
        #    super().__init__(*args, **kwargs)  # 원래 부모 init 호출
        #    self.apply_layers = apply_layers

        def _init_layers(self) -> None:
            """Initialize self_attn, cross-attn, ffn, and norms."""
            self.self_attn = MultiheadAttention(**self.self_attn_cfg)
            self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
            self.embed_dims = self.self_attn.embed_dims
            self.ffn = FFN(**self.ffn_cfg)

            self.proj_npy = nn.Linear(512,self.self_attn_cfg['embed_dims'])

            norms_list = [
                build_norm_layer(self.norm_cfg, self.embed_dims)[1]
                for _ in range(3)
            ]
            self.norms = ModuleList(norms_list)
        
        def forward(self,
                query: Tensor,
                npy_list: List,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                layer_id: Optional[int] = None,  # 🔥 추가
                apply_layers: Optional[List[int]] = None,  # 🔥 추가
                **kwargs) -> Tensor:
            """
            Args:
                query (Tensor): The input query, has shape (bs, num_queries, dim).
                key (Tensor, optional): The input key, has shape (bs, num_keys,
                    dim). If `None`, the `query` will be used. Defaults to `None`.
                value (Tensor, optional): The input value, has the same shape as
                    `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                    `key` will be used. Defaults to `None`.
                query_pos (Tensor, optional): The positional encoding for `query`,
                    has the same shape as `query`. If not `None`, it will be added
                    to `query` before forward function. Defaults to `None`.
                key_pos (Tensor, optional): The positional encoding for `key`, has
                    the same shape as `key`. If not `None`, it will be added to
                    `key` before forward function. If None, and `query_pos` has the
                    same shape as `key`, then `query_pos` will be used for
                    `key_pos`. Defaults to None.
                self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                    (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                    Defaults to None.
                cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                    (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                    Defaults to None.
                key_padding_mask (Tensor, optional): The `key_padding_mask` of
                    `self_attn` input. ByteTensor, has shape (bs, num_value).
                    Defaults to None.

            Returns:
                Tensor: forwarded results, has shape (bs, num_queries, dim).
            """
            
            '''
            A-Model implementation
            '''

            ''' only contrastive
            # 새로운 행과 열을 생성 (1행 1100열, 1100행 1열)
            if self_attn_mask != None:
                new_row = torch.zeros(1, query.shape[1], dtype=torch.bool).to(query.device)  # 새로운 첫 번째 행 (모두 False)
                new_col = torch.zeros(query.shape[1], 1, dtype=torch.bool).to(query.device)  # 새로운 첫 번째 열 (모두 False)
                new_corner = torch.tensor([[False]]).to(query.device)  # (1,1) 크기의 추가된 부분

                # 새로운 Self-Attention Mask 만들기
                extended_row = torch.cat([new_corner, new_row], dim=1)
                extended_self_attn_mask = torch.cat([new_col, self_attn_mask], dim=1)
                self_attn_mask = torch.cat([extended_row, extended_self_attn_mask], dim=0)

            # Query concat
            # npy_list를 PyTorch Tensor로 변환
            prompt_tensor = [torch.tensor(arr).unsqueeze(1) for arr in npy_list]  # 리스트의 NumPy 배열을 Tensor로 변환

            # 텐서를 dim=1 방향으로 concat (batch 차원 유지)
            concatenated_prompt = torch.cat(prompt_tensor, dim=0)
            concatenated_prompt = concatenated_prompt.to(query.device)
            concatenated_prompt = self.proj_npy(concatenated_prompt)

            zero_pos = torch.zeros_like(concatenated_prompt).to(query.device)

            # 원하는 방식으로 Concat 적용
            query = torch.cat([concatenated_prompt, query], dim=1)  # dim=1로 병합
            query_pos = torch.cat([zero_pos, query_pos], dim=1)
            '''

            query = self.self_attn(
                query=query,
                key=query,
                value=query,
                query_pos=query_pos,
                key_pos=query_pos,
                attn_mask=self_attn_mask, # None
                **kwargs)
            
            '''only contrastive
            query = query[:,1:,:]
            query_pos = query_pos[:,1:,:]
            '''

            #if self.training:
            self.saved_contrastive_features = []
            # if not hasattr(self, 'saved_conttrastive_features'):
            #     self.saved_contrastive_features = []
            self.saved_contrastive_features.append(query.detach())  # shape: (bs, num_queries, dim)
 
            # 🔥 apply_layers control
            if (apply_layers is not None) and (layer_id == apply_layers[-1]):
                # 만약 apply_layers의 마지막 layer_id와 현재 layer_id가 같다면
                # 여기까지만 계산하고 일찍 return
                del query, query_pos
                return

            query = self.norms[0](query)
            query = self.cross_attn(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_mask=cross_attn_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
            query = self.norms[1](query)
            query = self.ffn(query)
            query = self.norms[2](query)

            return query

class DeformableDetrTransformerDecoderLayer_OnlyContrastive_DINO(DetrTransformerDecoderLayer):
        """Decoder layer of Deformable DETR."""

        ### added code
        #def __init__(self, apply_layers=None, *args, **kwargs):
        #    super().__init__(*args, **kwargs)  # 원래 부모 init 호출
        #    self.apply_layers = apply_layers

        def _init_layers(self) -> None:
            """Initialize self_attn, cross-attn, ffn, and norms."""
            self.self_attn = MultiheadAttention(**self.self_attn_cfg)
            self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
            self.embed_dims = self.self_attn.embed_dims
            self.ffn = FFN(**self.ffn_cfg)

            self.proj_npy = nn.Linear(512,self.self_attn_cfg['embed_dims'])

            norms_list = [
                build_norm_layer(self.norm_cfg, self.embed_dims)[1]
                for _ in range(3)
            ]
            self.norms = ModuleList(norms_list)
        
        def forward(self,
                query: Tensor,
                npy_list: List,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                layer_id: Optional[int] = None,  # 🔥 추가
                apply_layers: Optional[List[int]] = None,  # 🔥 추가
                **kwargs) -> Tensor:
            """
            Args:
                query (Tensor): The input query, has shape (bs, num_queries, dim).
                key (Tensor, optional): The input key, has shape (bs, num_keys,
                    dim). If `None`, the `query` will be used. Defaults to `None`.
                value (Tensor, optional): The input value, has the same shape as
                    `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                    `key` will be used. Defaults to `None`.
                query_pos (Tensor, optional): The positional encoding for `query`,
                    has the same shape as `query`. If not `None`, it will be added
                    to `query` before forward function. Defaults to `None`.
                key_pos (Tensor, optional): The positional encoding for `key`, has
                    the same shape as `key`. If not `None`, it will be added to
                    `key` before forward function. If None, and `query_pos` has the
                    same shape as `key`, then `query_pos` will be used for
                    `key_pos`. Defaults to None.
                self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                    (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                    Defaults to None.
                cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                    (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                    Defaults to None.
                key_padding_mask (Tensor, optional): The `key_padding_mask` of
                    `self_attn` input. ByteTensor, has shape (bs, num_value).
                    Defaults to None.

            Returns:
                Tensor: forwarded results, has shape (bs, num_queries, dim).
            """
            
            '''
            A-Model implementation
            '''

            ''' only contrastive
            # 새로운 행과 열을 생성 (1행 1100열, 1100행 1열)
            if self_attn_mask != None:
                new_row = torch.zeros(1, query.shape[1], dtype=torch.bool).to(query.device)  # 새로운 첫 번째 행 (모두 False)
                new_col = torch.zeros(query.shape[1], 1, dtype=torch.bool).to(query.device)  # 새로운 첫 번째 열 (모두 False)
                new_corner = torch.tensor([[False]]).to(query.device)  # (1,1) 크기의 추가된 부분

                # 새로운 Self-Attention Mask 만들기
                extended_row = torch.cat([new_corner, new_row], dim=1)
                extended_self_attn_mask = torch.cat([new_col, self_attn_mask], dim=1)
                self_attn_mask = torch.cat([extended_row, extended_self_attn_mask], dim=0)

            # Query concat
            # npy_list를 PyTorch Tensor로 변환
            prompt_tensor = [torch.tensor(arr).unsqueeze(1) for arr in npy_list]  # 리스트의 NumPy 배열을 Tensor로 변환

            # 텐서를 dim=1 방향으로 concat (batch 차원 유지)
            concatenated_prompt = torch.cat(prompt_tensor, dim=0)
            concatenated_prompt = concatenated_prompt.to(query.device)
            concatenated_prompt = self.proj_npy(concatenated_prompt)

            zero_pos = torch.zeros_like(concatenated_prompt).to(query.device)

            # 원하는 방식으로 Concat 적용
            query = torch.cat([concatenated_prompt, query], dim=1)  # dim=1로 병합
            query_pos = torch.cat([zero_pos, query_pos], dim=1)
            '''

            query = self.self_attn(
                query=query,
                key=query,
                value=query,
                query_pos=query_pos,
                key_pos=query_pos,
                attn_mask=self_attn_mask, # None
                **kwargs)
            
            '''only contrastive
            query = query[:,1:,:]
            query_pos = query_pos[:,1:,:]
            '''

            #if self.training:
            self.saved_contrastive_features = []
            # if not hasattr(self, 'saved_conttrastive_features'):
            #     self.saved_contrastive_features = []
            self.saved_contrastive_features.append(query.detach())  # shape: (bs, num_queries, dim)
 
            # 🔥 apply_layers control
            if (apply_layers is not None) and (layer_id == apply_layers[-1] + 1):
                # 만약 apply_layers의 마지막 layer_id와 현재 layer_id가 같다면
                # 여기까지만 계산하고 일찍 return
                del query, query_pos
                return

            query = self.norms[0](query)
            query = self.cross_attn(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_mask=cross_attn_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
            query = self.norms[1](query)
            query = self.ffn(query)
            query = self.norms[2](query)

            return query

class DeformableDetrTransformerDecoderLayer_ContrastiveMOCA_DINO(DetrTransformerDecoderLayer):
        """Decoder layer of Deformable DETR."""

        ### added code
        #def __init__(self, apply_layers=None, *args, **kwargs):
        #    super().__init__(*args, **kwargs)  # 원래 부모 init 호출
        #    self.apply_layers = apply_layers

        def _init_layers(self) -> None:
            """Initialize self_attn, cross-attn, ffn, and norms."""
            self.self_attn = MultiheadAttention(**self.self_attn_cfg)
            self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
            self.embed_dims = self.self_attn.embed_dims
            self.ffn = FFN(**self.ffn_cfg)

            self.proj_npy = nn.Linear(512,self.self_attn_cfg['embed_dims'])

            norms_list = [
                build_norm_layer(self.norm_cfg, self.embed_dims)[1]
                for _ in range(3)
            ]
            self.norms = ModuleList(norms_list)
        
        def forward(self,
                query: Tensor,
                npy_list: List,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                layer_id: Optional[int] = None,  # 🔥 추가
                apply_layers: Optional[List[int]] = None,  # 🔥 추가
                **kwargs) -> Tensor:
            """
            Args:
                query (Tensor): The input query, has shape (bs, num_queries, dim).
                key (Tensor, optional): The input key, has shape (bs, num_keys,
                    dim). If `None`, the `query` will be used. Defaults to `None`.
                value (Tensor, optional): The input value, has the same shape as
                    `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                    `key` will be used. Defaults to `None`.
                query_pos (Tensor, optional): The positional encoding for `query`,
                    has the same shape as `query`. If not `None`, it will be added
                    to `query` before forward function. Defaults to `None`.
                key_pos (Tensor, optional): The positional encoding for `key`, has
                    the same shape as `key`. If not `None`, it will be added to
                    `key` before forward function. If None, and `query_pos` has the
                    same shape as `key`, then `query_pos` will be used for
                    `key_pos`. Defaults to None.
                self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                    (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                    Defaults to None.
                cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                    (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                    Defaults to None.
                key_padding_mask (Tensor, optional): The `key_padding_mask` of
                    `self_attn` input. ByteTensor, has shape (bs, num_value).
                    Defaults to None.

            Returns:
                Tensor: forwarded results, has shape (bs, num_queries, dim).
            """
            '''
            A-Model implementation
            '''

            # 새로운 행과 열을 생성 (1행 1100열, 1100행 1열)
            if self_attn_mask != None:
                new_row = torch.zeros(1, query.shape[1], dtype=torch.bool).to(query.device)  # 새로운 첫 번째 행 (모두 False)
                new_col = torch.zeros(query.shape[1], 1, dtype=torch.bool).to(query.device)  # 새로운 첫 번째 열 (모두 False)
                new_corner = torch.tensor([[False]]).to(query.device)  # (1,1) 크기의 추가된 부분

                # 새로운 Self-Attention Mask 만들기
                extended_row = torch.cat([new_corner, new_row], dim=1)
                extended_self_attn_mask = torch.cat([new_col, self_attn_mask], dim=1)
                self_attn_mask = torch.cat([extended_row, extended_self_attn_mask], dim=0)

            # Query concat
            # npy_list를 PyTorch Tensor로 변환
            prompt_tensor = [torch.tensor(arr[1]).unsqueeze(1) for arr in npy_list]  # 리스트의 NumPy 배열을 Tensor로 변환

            # 텐서를 dim=1 방향으로 concat (batch 차원 유지)
            concatenated_prompt = torch.cat(prompt_tensor, dim=0)
            concatenated_prompt = concatenated_prompt.to(query.device)
            concatenated_prompt = self.proj_npy(concatenated_prompt)

            zero_pos = torch.zeros_like(concatenated_prompt).to(query.device)

            # 원하는 방식으로 Concat 적용
            query = torch.cat([concatenated_prompt, query], dim=1)  # dim=1로 병합
            query_pos = torch.cat([zero_pos, query_pos], dim=1)


            query = self.self_attn(
                query=query,
                key=query,
                value=query,
                query_pos=query_pos,
                key_pos=query_pos,
                attn_mask=self_attn_mask, # None
                **kwargs)
            
            query = query[:,1:,:]
            query_pos = query_pos[:,1:,:]

            #if self.training:
            self.saved_contrastive_features = []
            # if not hasattr(self, 'saved_conttrastive_features'):
            #     self.saved_contrastive_features = []
            self.saved_contrastive_features.append(query.detach())  # shape: (bs, num_queries, dim)
 
            # 🔥 apply_layers control
            if (apply_layers is not None) and (layer_id == apply_layers[-1] + 1) :
                # 만약 apply_layers의 마지막 layer_id와 현재 layer_id가 같다면
                # 여기까지만 계산하고 일찍 return
                del query, query_pos
                return

            query = self.norms[0](query)
            query = self.cross_attn(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_mask=cross_attn_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
            query = self.norms[1](query)
            query = self.ffn(query)
            query = self.norms[2](query)

            return query
