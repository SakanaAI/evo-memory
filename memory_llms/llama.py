import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any, List
from transformers.cache_utils import Cache, DynamicCache, StaticCache


import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb, repeat_kv, 
    LlamaForCausalLM, 
    LlamaPreTrainedModel,
    LlamaAttention, 
    LlamaRotaryEmbedding, 
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaModel,
    AttentionMaskConverter,
    )
from memory_policy import MemoryPolicy
from utils import empty_gpu_cache, get_all_submodules
from memory_llms.base import (
    MemoryModelWrapper, MemoryAttention, MemoryDecoderLayer)

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

@dataclass
class MemoryModelOutputWithPast(BaseModelOutputWithPast):
    query_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    position_ids: Optional[torch.LongTensor] = None

@dataclass
class CausalMemoryLMOutputWithPast(CausalLMOutputWithPast):
    query_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    position_ids: Optional[torch.LongTensor] = None

class LlamaNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """NTK-aware, alsways rescaling by the MAX. """

    def __init__(self, dim, max_position_embeddings=2048, base=10000,
                 device=None, scaling_factor=1.0, alpha=2):
        super().__init__(
            dim, max_position_embeddings, base, device, scaling_factor)
        self.manual_max_pi = max_position_embeddings

        self.alpha = alpha
        assert alpha >= 1

        self.base = self.base * ((self.alpha*self.scaling_factor) - (
            self.alpha - 1))**(self.dim/(self.dim - 2))
        inv_freq = 1.0 / (self.base**(torch.arange(
            0, self.dim, 2, dtype=torch.int64).float()/self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False) 
    
class LlamaDynamicNTKScalingRotaryEmbeddingMax(LlamaRotaryEmbedding):
    """NTK-aware, always rescaling by the MAX using the scaling factor 
       (Extra-conservative) """
    def forward(self, x, position_ids):
        seq_len = self.max_position_embeddings*self.scaling_factor
        if seq_len > self.max_position_embeddings:
            base = self.base * ((
                self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            )**(self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base**(torch.arange(
                0, self.dim, 2, dtype=torch.int64).float().to(
                    x.device) / self.dim))
            
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        cos, sin = super().forward(x, position_ids)
        return cos, sin

class LlamaDynamicNTKScalingRotaryEmbeddingManual(LlamaRotaryEmbedding):
    """Requires manually setting max_seq_len for generation"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, 
                 device=None, scaling_factor=1):
        super().__init__(dim, max_position_embeddings, base, device, 
                         scaling_factor)
        self.max_seq_lens = None
    
    def set_max_seq_lens(self, max_seq_lens: Optional[torch.tensor]):
        if max_seq_lens is not None:
            max_seq_lens = max_seq_lens.to(
                dtype=torch.long, device=self.inv_freq.device).view(-1, 1)
        self.max_seq_lens = max_seq_lens

    @torch.no_grad()
    def forward(self, x, position_ids):

        if self.max_seq_lens is None:
            max_seq_lens = torch.max(position_ids, dim=-1, keepdim=True).values
        else:
            max_seq_lens = self.max_seq_lens
        
        use_extended_freqs = max_seq_lens > self.max_position_embeddings

        extended_base = self.base*(
            (self.scaling_factor*max_seq_lens/self.max_position_embeddings) - (
                self.scaling_factor - 1))**(self.dim / (self.dim - 2))
        
        extended_inv_freq = 1.0/extended_base**(torch.arange(
            0, self.dim, 2, dtype=torch.int64).float().to(x.device)/self.dim)
        
        inv_freq = torch.where(use_extended_freqs, extended_inv_freq, 
                               self.inv_freq.unsqueeze(0))
        # bs x dim//2 x 1
        inv_freq_expanded = inv_freq[:, :, None].float().expand(
            position_ids.shape[0], -1, 1)
        
        # bs x 1 x seq_lens
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(
            device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ 
                     position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        
class WrappedLlamaForCausalLM(LlamaForCausalLM, MemoryModelWrapper):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, model: LlamaForCausalLM, 
                 memory_policy: MemoryPolicy,
                 update_cache_frequency: int = 1, 
                 override_attn: bool = True,
                 max_new_tokens: Optional[int] = None,
                 memory_policy_fixed_delay: Optional[int] = None,
                 output_attentions_full_precision: bool = True,
                 ):
        self.config: LlamaConfig = copy.deepcopy(model.config)

        LlamaPreTrainedModel.__init__(self, self.config)

        self.model = LlamaMemoryModel(self.config)
        self.memory_policy = memory_policy
        self.max_new_tokens = max_new_tokens

        checkpoint = copy.deepcopy(model.state_dict())

        del model
        empty_gpu_cache()

        self.base_model_param_keys = list(checkpoint.keys())
        self.registration_kwargs = dict(
            num_heads=self.config.num_key_value_heads)
        
        self.max_seq_lens: Optional[torch.Tensor] = None
    
        self.vocab_size = self.config.vocab_size
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False)
        
        if override_attn:
            self.load_partial_state_dict(checkpoint)

        self.registration_kwargs = dict(
            num_heads=self.config.num_key_value_heads,
            num_key_value_groups=(self.config.num_attention_heads//
                                  self.config.num_key_value_heads),
            num_memory_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            )
        
        MemoryModelWrapper.__init__(
            self,
            config=self.config,
            memory_policy=memory_policy,
            registration_kwargs=self.registration_kwargs,
            memory_policy_fixed_delay=memory_policy_fixed_delay,
            output_attentions_full_precision=output_attentions_full_precision,
            )
        
        self.memory_requires_attn = self.memory_policy.requires_attn_scores
        self.memory_requires_queries = self.memory_policy.requires_queries

        self.update_cache_frequency = update_cache_frequency
        if self.update_cache_frequency > 1:
            raise NotImplementedError
        
        self.attn_layers = get_all_submodules(module=self.model, 
                                              target_layer_class=LlamaAttention,
                                              include_subclasses=True)
        print(f'Instantiating memory llm with {len(self.attn_layers)} ' +
              'attention layers')
        
    
    def move_model_to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        self.lm_head.to(*args, **kwargs)
        return self


    def set_max_seq_lens(self, max_seq_lens):
        for l in self.attn_layers:
            l.set_max_seq_lens(max_seq_lens)
        self.max_seq_lens = max_seq_lens

    def load_cached_attn_mxs(self, cached_attn_mxs):
        self.memory_policy.load_cached_attn_mxs(
            cached_attn_mxs=cached_attn_mxs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_queries: bool = False,
        apply_memory_policy: bool = True,
        limit_new_tokens: Optional[int] = None,
        output_attentions_full_precision: Optional[bool] = None,
        memory_policy_kwargs: Optional[dict] = None,
        # used for analysis purposes, skips application of memory model
        always_buffer_cache: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if memory_policy_kwargs is None:
            memory_policy_kwargs = {}
        
        output_attentions = (output_attentions if output_attentions is not None 
                             else self.config.output_attentions)
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
            )
        
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)

        output_attentions_full_precision = (
            output_attentions_full_precision if output_attentions_full_precision
            is not None else self.output_attentions_full_precision
            )

        if input_ids is not None:
            bs, num_new_tokens = input_ids.shape[:2]
            device = input_ids.device
        else:
            bs, num_new_tokens = inputs_embeds.shape[:2]
            device = inputs_embeds.device

        if limit_new_tokens is None:
            limit_new_tokens = self.max_new_tokens

        if position_ids is None:

            if attention_mask is not None:
                position_ids = torch.cumsum(attention_mask[-num_new_tokens:],
                                            dim=-1) - 1
            else:
                position_ids = torch.arange(
                    num_new_tokens, device=device).unsqueeze(0).expand(bs, -1)
        
        num_new_tokens = input_ids.shape[-1]

        split_processing = False
        
        if self.memory_policy_fixed_delay is not None and num_new_tokens > 1:
            past_length = 0
            if past_key_values is not None:
                num_all_tokens = past_key_values[0][0].shape[-2]
                if num_all_tokens > 0:
                    past_length = self.memory_policy.get_rotary_offset(
                        layer_id=0).item()
                
            split_frequency = self.memory_policy_fixed_delay  
            if limit_new_tokens is not None:
                assert split_frequency % limit_new_tokens == 0  
                split_frequency = limit_new_tokens
                
            new_length = past_length + num_new_tokens
            past_length_residual = past_length % split_frequency

            
            next_break_point_idx = (
                split_frequency - past_length_residual)
            if next_break_point_idx < num_new_tokens:
                split_idxs = np.arange(next_break_point_idx, num_new_tokens,
                                       split_frequency)
                split_idxs = np.append(split_idxs, num_new_tokens)
                split_lens = np.copy(split_idxs)
                split_lens[1:] = np.diff(split_idxs)
                split_processing = True
        elif limit_new_tokens is not None:
            if num_new_tokens > limit_new_tokens:
                split_processing = True
                split_idxs = np.arange(
                    limit_new_tokens, num_new_tokens, limit_new_tokens)
                split_idxs = np.append(split_idxs, num_new_tokens)
                split_lens = np.copy(split_idxs)
                split_lens[1:] = np.diff(split_idxs)        
        
        # split input context to apply memory model and reduce GPU memory
        if split_processing:
            split_idxs = split_idxs.tolist()
            split_lens = split_lens.tolist()
            assert labels is None, (
              'Tensor splitting has not been tested for training')
            unspecified_max_seq_lens = self.max_seq_lens is None
            if unspecified_max_seq_lens:
                max_position_ids = position_ids.max(
                    dim=-1, keepdim=True).values
                self.set_max_seq_lens(max_position_ids)
            
            split_tokens = torch.split(
                input_ids,
                split_size_or_sections=split_lens,
                dim=-1,
                )

            split_position_ids = torch.split(
                position_ids,
                split_size_or_sections=split_lens,
                dim=-1,
                )
            
            if cache_position is not None:
                split_cache_position = torch.split(
                    cache_position,
                    split_size_or_sections=split_lens,
                    dim=-1,
                    )
                
            num_splits = len(split_tokens)

            for idx in range(num_splits-1):
                input_ids = split_tokens[idx]

                position_ids = split_position_ids[idx]
                if attention_mask is not None:
                    curr_attn_mask = attention_mask[..., :split_idxs[idx]]
                else:
                    curr_attn_mask = None

                if cache_position is not None:
                    cache_position = split_cache_position[idx]

                curr_outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=curr_attn_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                    apply_memory_policy=apply_memory_policy,
                    limit_new_tokens=None,
                    output_attentions_full_precision=(
                        output_attentions_full_precision),
                    always_buffer_cache=always_buffer_cache,
                )
                past_key_values = curr_outputs.past_key_values
                
            input_ids = split_tokens[-1]
            position_ids = split_position_ids[-1]
            if cache_position is not None:
                cache_position = split_cache_position[-1]
        
            num_new_tokens = input_ids.shape[1]

            if unspecified_max_seq_lens:
                self.set_max_seq_lens(None)

        assert use_cache, 'Make sure the KV cache is being used'

        output_attentions = (
            output_attentions or self.memory_policy.requires_attn_scores)
        
        output_queries = (
            output_queries or self.memory_policy.requires_queries)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            output_attentions_full_precision=output_attentions_full_precision,
        )


        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) 
                      for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if apply_memory_policy:
            update_iter = True
            if self.memory_policy_fixed_delay is not None:
                if past_key_values is not None:
                    past_length = self.memory_policy.get_rotary_offset(
                        layer_id=0).item()
                else:
                    past_length = 0

                new_length = past_length + num_new_tokens
                
                if not (new_length % self.memory_policy_fixed_delay == 0):
                    update_iter = False
                
            
            # cache is automatically converted to 'legacy_cache' 
            # tuple of tuples (num_layers x 2) after model inference
            if update_iter and (not always_buffer_cache):
                outputs.past_key_values = self.memory_policy.update_cache(
                    past_key_values=outputs.past_key_values, 
                    num_new_tokens=num_new_tokens,
                    attn_weights_list=outputs.attentions,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **memory_policy_kwargs,
                    )
            elif not update_iter:
                self.memory_policy.buffer_cache(
                    past_key_values=outputs.past_key_values, 
                    num_new_tokens=num_new_tokens,
                    attn_weights_list=outputs.attentions,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    )

        else:
            self.memory_policy.update_rotary_offset(
                num_new_tokens=num_new_tokens,
                num_all_tokens=outputs.past_key_values[0][0].shape[-2]
                )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalMemoryLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            query_states=outputs.query_states,
            position_ids=position_ids,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs = None,
        generation_config = None,
        logits_processor = None,
        stopping_criteria = None,
        prefix_allowed_tokens_fn = None,
        synced_gpus = None,
        assistant_model = None,
        streamer = None,
        negative_prompt_ids = None,
        negative_prompt_attention_mask = None,
        **kwargs,):
        
        generated_output = super().generate(
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            assistant_model,
            streamer,
            negative_prompt_ids,
            negative_prompt_attention_mask,
            **kwargs,)

        return generated_output


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, 
        inputs_embeds=None, cache_position=None, **kwargs,
    ):
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(getattr(self.model.layers[0], "self_attn",
                                              {}), "past_key_value", None)
            has_static_cache = past_key_values is not None
        
           
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        past_length = 0
        if past_key_values is not None:
            past_length = self.memory_policy.get_rotary_offset(layer_id=0)
            if (self.cache_size is not None 
                and self.memory_policy_fixed_delay is None):
                max_cache_length = torch.tensor(self.cache_size, 
                                                device=past_length.device)
            else:
                max_cache_length = None

            if max_cache_length is None:
                cache_length = past_length
            else:
                cache_length = torch.min(max_cache_length, past_length)

            if (attention_mask is not None and
                attention_mask.shape[1] > input_ids.shape[1]):
                input_ids = input_ids[
                    :, -(attention_mask.shape[1] - past_length) :]


            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            position_ids = position_ids[:, -input_ids.shape[1] :]
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, 
                    -(max_cache_length + input_ids.shape[1]):]


        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}
        input_length = (position_ids.shape[-1] if position_ids is not None
                        else input_ids.shape[-1])
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + input_length,
                device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "apply_memory_policy": kwargs.get("apply_memory_policy", True),
                "limit_new_tokens": kwargs.get("limit_new_tokens", None),
            }
        )
        return model_inputs
    
class LlamaMemoryAttention(LlamaAttention, MemoryAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        output_queries: bool = False,
        output_attentions_full_precision: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * 
                                 self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) //
                self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) 
                            for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) 
                          for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) 
                            for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos,
                            "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is None:
            n_q, n_k = attn_weights.shape[-2:]
            dtype, device = hidden_states.dtype, hidden_states.device
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full((n_q, n_q), fill_value=min_dtype, 
                                     dtype=dtype, device=device)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = F.pad(causal_mask, (n_k-n_q, 0))
            attn_weights = attn_weights + causal_mask
        else:
            causal_mask = attention_mask[:, :, :, -key_states.shape[-2]:]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights_to_return = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_to_return.to(query_states.dtype)

        if not output_attentions_full_precision:
            attn_weights_to_return = attn_weights

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError()
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum(
                [F.linear(attn_output[i], o_proj_slices[i])
                 for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
            attn_weights_to_return = None
        
        if not output_queries:
            return attn_output, attn_weights_to_return, past_key_value
        else:
            return (attn_output, attn_weights_to_return, past_key_value,
                    query_states)
    
    def set_max_seq_lens(self, max_seq_lens: Optional[int]):
        if self.require_manual_max_seq_lens:
            self.rotary_emb.set_max_seq_lens(max_seq_lens)

    def _init_rope(self):
        self.require_manual_max_seq_lens = False

        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if 'override' in self.config.rope_scaling:
                scaling_type = self.config.rope_scaling["override"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type.lower() == "ntk":
                scaling_alpha = self.config.rope_scaling["alpha"]
                self.rotary_emb = LlamaNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    alpha=scaling_alpha,
                )
            elif scaling_type == "dynamic_manual":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbeddingManual(
                        self.head_dim,
                        max_position_embeddings=self.max_position_embeddings,
                        scaling_factor=scaling_factor,
                        base=self.rope_theta,
                        )
                self.require_manual_max_seq_lens = True
            elif scaling_type == "dynamic":
                assert 'fix' in self.config.rope_scaling, (
                    'dynamic scaling + kv cache is bugged')
                dynamic_fix = self.config.rope_scaling['fix']
                if dynamic_fix == 'max':
                    self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbeddingMax(
                        self.head_dim,
                        max_position_embeddings=self.max_position_embeddings,
                        scaling_factor=scaling_factor,
                        base=self.rope_theta,
                    )
                elif dynamic_fix == 'manual':
                    self.rotary_emb = (
                        LlamaDynamicNTKScalingRotaryEmbeddingManual(
                        self.head_dim,
                        max_position_embeddings=self.max_position_embeddings,
                        scaling_factor=scaling_factor,
                        base=self.rope_theta,
                        )
                    )
                    self.require_manual_max_seq_lens = True
                elif dynamic_fix == 'none':
                    print('WARNING: HF dynamic scaling + kv cache is bugged,' +
                          ' no fix is being applied...')
                    self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                        self.head_dim,
                        max_position_embeddings=self.max_position_embeddings,
                        scaling_factor=scaling_factor,
                        base=self.rope_theta,
                    )
                else:
                    raise ValueError(
                        f"Unknown NTK RoPE scaling fix {dynamic_fix}")
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

class LlamaMemoryDecoderLayer(nn.Module, MemoryDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaMemoryAttention(
            config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        output_queries: bool = False,
        output_attentions_full_precision: bool = True,
        **kwargs,
    ) -> Tuple[torch.FloatTensor,
               Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if output_queries:
                (hidden_states, self_attn_weights, present_key_value,
                 query_states) = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    output_queries=output_queries,
                    output_attentions_full_precision=(
                        output_attentions_full_precision),
                    **kwargs,
            )
        else:
            hidden_states, self_attn_weights, present_key_value = (
                self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    output_queries=output_queries,
                    output_attentions_full_precision=(
                        output_attentions_full_precision),
                    **kwargs,
            ))
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        
        if output_queries:
            outputs += (query_states,)

        return outputs

class LlamaMemoryModel(LlamaModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaMemoryDecoderLayer(config, layer_idx) for layer_idx
             in range(config.num_hidden_layers)])
        
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_queries: bool = False,
        output_attentions_full_precision: bool = True,

    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (use_cache if use_cache is not None
                     else self.config.use_cache)
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)


        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the " +
                "same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = (past_key_values.get_seq_length() 
                                if past_key_values is not None else 0)
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
                  device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values,
            output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        all_query_states = () if output_queries else None

        if use_cache:
            cache_idx = 2 if output_attentions else 1
            query_idx = cache_idx + 1
        else:
            query_idx = 2 if output_attentions else 1

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    output_queries,
                    output_attentions_full_precision=(
                        output_attentions_full_precision),
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    output_queries=output_queries,
                    output_attentions_full_precision=(
                        output_attentions_full_precision),
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[cache_idx]

            if output_queries:
                all_query_states += (layer_outputs[query_idx],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, 
                            all_self_attns, all_query_states] if v is not None)
        return MemoryModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            query_states=all_query_states,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = (past_key_values.get_seq_length() 
                            if past_key_values is not None else 0)
        using_static_cache = isinstance(past_key_values, StaticCache)

        if (self.config._attn_implementation == "sdpa" and 
            not using_static_cache and not output_attentions):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed " +
                                 "in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), 
                fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(
                input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[
                    :, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype)

        return causal_mask
