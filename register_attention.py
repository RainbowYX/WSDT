# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from diffusers import StableDiffusionXLPipeline
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from diffusers.models import attention_processor
from tqdm import tqdm
T = torch.Tensor

def expand_first(feat: T) -> T:
    b = feat.shape[0]
    feat_style = feat[0].unsqueeze(0)
    feat_style = feat_style.repeat(b,1,1)
    return feat_style.reshape(*feat.shape)

def adain(feat: T, c_dim: int=1, threshold: float = None, eps: float = 1e-5) -> T:
    if c_dim == 2:
        b,c,h,w=feat.shape
        feat = feat.reshape(b,c,h*w)

    feat_mean = feat.mean(dim=c_dim, keepdims=True)
    feat_std = (feat.var(dim=c_dim, keepdims=True) + eps).sqrt()
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)

    mean_d=(feat_mean - feat_style_mean) ** 2 
    std_d=(feat_std - feat_style_std) ** 2 
    distance=mean_d+std_d

    if threshold:
        feat_style_mean=torch.where(distance>threshold,feat_mean,feat_style_mean)
        feat_style_std=torch.where(distance>threshold,feat_std,feat_style_std)

    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean

    if c_dim == 2:
        feat = feat.reshape(b,c,h,w)

    return feat

def _encode_image(model: StableDiffusionXLPipeline, image: np.ndarray) -> T:
    model.vae.to(dtype=torch.float32)
    image = torch.from_numpy(image).float() / 255.
    image = (image * 2 - 1).permute(2, 0, 1).unsqueeze(0)
    latent = model.vae.encode(image.to(model.vae.device))['latent_dist'].mean * model.vae.config.scaling_factor
    model.vae.to(dtype=torch.float16)
    return latent

def make_inversion_callback(zts):
    def callback_on_step_end(pipeline: StableDiffusionXLPipeline, i: int, t: T, callback_kwargs: dict[str, T]) -> dict[str, T]:
        latents = callback_kwargs['latents']
        if i<50:
            latents[0] = zts[i+1].to(latents.device, latents.dtype)
        latents=adain(latents,2)
        return {'latents': latents}
    return  callback_on_step_end

@torch.no_grad()
def forward_noisy(model: StableDiffusionXLPipeline, x0: np.ndarray, num_inference_steps: int):
    z0 = _encode_image(model, x0)
    model.scheduler.set_timesteps(num_inference_steps, device=z0.device)
    forward_latent = [z0]
    noise = torch.randn(z0.shape, generator=torch.Generator(device="cuda").manual_seed(0), device=z0.device, dtype=z0.dtype)

    for i in tqdm(range(model.scheduler.num_inference_steps)):
        t = model.scheduler.timesteps[len(model.scheduler.timesteps) - i - 1]
        latent = model.scheduler.add_noise(z0, noise, t)
        forward_latent.append(latent)
    return torch.cat(forward_latent).flip(0)

class DefaultAttentionProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = attention_processor.AttnProcessor2_0()
    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        return self.processor(attn, hidden_states, encoder_hidden_states, attention_mask)

class SelfAttentionProcessor(DefaultAttentionProcessor):
    def register_call(
            self,
            attn: attention_processor.Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            **kwargs
    ):  
        hidden_states=adain(hidden_states,1,self.threshold)
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
                
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        value=adain(value,1,self.threshold)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = nnf.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states=adain(hidden_states,1,self.threshold)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        hidden_states = self.register_call(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
        return hidden_states

    def __init__(self, threshold):
        super().__init__()
        self.threshold=threshold


def register_attention_processors(pipeline: StableDiffusionXLPipeline, threshold: float = 0.2):
    attn_procs = {}
    unet = pipeline.unet

    for i, name in enumerate(unet.attn_processors.keys()):
        if 'attn1' in name and 'up_blocks' in name:
            attn_procs[name] = SelfAttentionProcessor(threshold)
        else:
            attn_procs[name] = DefaultAttentionProcessor()

    unet.set_attn_processor(attn_procs)

