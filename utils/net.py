#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import timm
import os
from typing import Any, Callable, Dict, Optional, Union
import logging
try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

_logger = logging.getLogger(__name__)

def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def load_state_dict(
        checkpoint_path: str,
        use_ema: bool = True,
        device: Union[str, torch.device] = 'cpu',
) -> Dict[str, Any]:
    if checkpoint_path and os.path.isfile(checkpoint_path):
        # Check if safetensors or not and load weights accordingly
        if str(checkpoint_path).endswith(".safetensors"):
            assert _has_safetensors, "`pip install safetensors` to use .safetensors"
            checkpoint = safetensors.torch.load_file(checkpoint_path, device=device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        state_dict = clean_state_dict(checkpoint[state_dict_key] if state_dict_key else checkpoint)
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()
def remap_state_dict(
        state_dict: Dict[str, Any],
        model: torch.nn.Module,
        allow_reshape: bool = True
):
    """ remap checkpoint by iterating over state dicts in order (ignoring original keys).
    This assumes models (and originating state dict) were created with params registered in same order.
    """
    out_dict = {}
    for (ka, va), (kb, vb) in zip(model.state_dict().items(), state_dict.items()):
        assert va.numel() == vb.numel(), f'Tensor size mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed.'
        if va.shape != vb.shape:
            if allow_reshape:
                vb = vb.reshape(va.shape)
            else:
                assert False,  f'Tensor shape mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed.'
        out_dict[ka] = vb
    return out_dict

def load_checkpoint(
        model: torch.nn.Module,
        checkpoint_path: str,
        use_ema: bool = True,
        device: Union[str, torch.device] = 'cpu',
        strict: bool = True,
        remap: bool = True,
        filter_fn: Optional[Callable] = None,
):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return

    state_dict = load_state_dict(checkpoint_path, use_ema, device=device)
    if remap:
        state_dict = remap_state_dict(state_dict, model)
    elif filter_fn:
        state_dict = filter_fn(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


class classifier(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, embedding_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_classes)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
    def forward(self, x: torch.Tensor):
        # <--- 优化点: 使用 torch.mean 实现全局平均池化
        # dim=(-2, -1) 表示对最后两个维度 (H, W) 求均值
        # keepdim=True 保持维度为 [B, C, 1, 1]，以便后续 flatten
        x = torch.mean(x, dim=(-2, -1), keepdim=True)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        feature = F.gelu(x)
        out = self.fc2(feature)
        return feature, out


class Net(nn.Module):
    def __init__(self,model_name ,num_class,mode='train',embeddingdim=512):
        super(Net, self).__init__()
        self.backbone = timm.create_model(model_name,features_only=True, out_indices=[-1],pretrained=False )

        if('dinov3_lvd1689m') in model_name:
            filename = model_name.split('.')[0]+'_dinov3'
        else:
            filename=model_name.split('.')[0]

        if('mambaout') in model_name:  #mambaout 系列的输出为（B,H,W,C）
            self.channel_last=True
        else:
            self.channel_last=False

        try :
            load_checkpoint(self.backbone, './premodels/{}.pth'.format(filename))
        except:

            load_checkpoint(self.backbone, './utils/premodels/{}.pth'.format(filename))


        # aa=list(selected_feature_extractor.named_modules())[-1]
        # ee=list(model.named_modules())[-1][1]
        # if ('dinov3' in model_name):
        #     if ('convnext_tiny.dinov3_lvd1689m') in model_name:
        #         fc_in_ch = list(selected_feature_extractor.named_modules())[233][1].out_features
        #
        #     else:
        #         fc_in_ch = list(selected_feature_extractor.named_modules())[246][1].out_features
        #
        # else:
        #     fc_in_ch = list(selected_feature_extractor.named_modules())[161][1].out_channels

        fc_in_ch = self.backbone.feature_info[-1]['num_chs']
        self.classifier = classifier(fc_in_ch, num_class,embeddingdim)
        self.mode=mode

    def forward(self, x):
        x = self.backbone(x)[0]
        if self.channel_last:
            x = x.permute((0, 3, 1, 2)).contiguous()
        feature,out= self.classifier(x)
        if self.mode=='train':
            return feature,out
        elif self.mode=='eval':
            return feature, out
        else:
            softmax_result=torch.softmax(out, dim=1)
            score,index=torch.max( softmax_result, dim=1)

            index = torch.tensor(index, dtype=torch.float32)
            score=torch.tensor(score, dtype=torch.float32)

            return index,score

if __name__=="__main__":
    # net=Net('convnext_pico.d1_in1k',num_class=9,embeddingdim=128,mode='pred')
    # net = Net('convnext_tiny.dinov3_lvd1689m', num_class=9, embeddingdim=128, mode='pred')
    # net = Net('vit_base_patch16_dinov3.lvd_1689m', num_class=9, embeddingdim=128, mode='pred')
    # net = Net('fastvit_mci3.apple_mclip2_dfndr2b', num_class=9, embeddingdim=128, mode='pred')
    # net = Net('naflexvit_base_patch16_siglip.v2_webli', num_class=9, embeddingdim=128, mode='pred')
    # net = Net('fasternet_t0.in1k', num_class=9, embeddingdim=128, mode='pred')
    # net = Net('rdnet_small.nv_in1k', num_class=9, embeddingdim=128, mode='pred')
    # net = Net('mambaout_femto.in1k', num_class=9, embeddingdim=128, mode='pred')
    net = Net('tf_efficientnet_b1.ns_jft_in1k', num_class=9, embeddingdim=128, mode='pred')
    # summary(net,(3,224,224))
    net.eval()
    in_ten = torch.randn(3, 3,224, 224)
    index,score=net(in_ten)
    print(index)
    print(score)