import gc
import inspect
from torch.nn import functional as F

import collections

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)
import torch
import json
import copy

import numpy as np

import hydra
from omegaconf import DictConfig

from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval.models.utils import Collator
from transformers.activations import ACT2FN


BYTES_TO_MB = 1024**2


def get_gpu_memory_mb():
    return torch.cuda.memory_allocated()/BYTES_TO_MB


def get_peak_gpu_memory_allocated_mb():
    return torch.cuda.max_memory_allocated()/BYTES_TO_MB


def reset_peak_gpu_memory_stats():
    max_allocated = get_peak_gpu_memory_allocated_mb()
    torch.cuda.reset_peak_memory_stats(device=None)
    return max_allocated


def get_nonlinearity(
        nonlinearity: Optional[Union[str, Callable]],
) -> Callable:
    if nonlinearity is None:
        def id_fn(x):
            return x
        return id_fn
    elif isinstance(nonlinearity, Callable):
        return nonlinearity
    elif nonlinearity.lower() in ACT2FN:
        return ACT2FN[nonlinearity.lower()]
    else:
        raise NotImplementedError


def reconstruct_causal_mask(n_q, n_k,
                            # bs x 1 x n_k
                            attn_mask):
    device = attn_mask.device
    dtype = attn_mask.dtype
    causal_mask = torch.ones((n_q, n_q), dtype=dtype, device=device)
    causal_mask = torch.tril(causal_mask, diagonal=0)

    causal_mask = F.pad(causal_mask, (n_k-n_q, 0), value=1.0)

    # bs x 1 x n_q, n_k
    mask = causal_mask*attn_mask.unsqueeze(-2)
    return mask


def num_attending_queries(n_q, n_k, attn_mask):
    '''determines number of attending queries for each key, applying
       causal ordering'''
    # bs x 1 x n_q, n_k
    mask = reconstruct_causal_mask(n_q=n_q, n_k=n_k, attn_mask=attn_mask)
    # bs x 1 x n_k
    return torch.sum(mask, dim=-2)


def safe_tensor_print(tensor, limit=3):
    if isinstance(tensor, torch.Tensor):
        print(tensor.squeeze()[:limit])
    else:
        print(tensor)


def empty_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()


def get_first_subseq_split(seq, subseq):
    n = len(seq)
    m = len(subseq)
    for i in range(n - m + 1):
        found = True
        for j in range(m):
            if seq[i + j] != subseq[j]:
                found = False
                break
        if found:
            return seq[:i]
    return seq


def get_first_value_split(seq, value):
    n = len(seq)
    for i in range(n):
        if seq[i] == value:
            return seq[:i]
    return seq


def is_oom_exception(exception: Exception) -> bool:
    # based on accelerate library
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


def get_all_submodules(module, target_layer_class, include_subclasses=True):
    target_submodules = []
    for name, sub_module in reversed(module._modules.items()):
        if include_subclasses:
            is_target = isinstance(sub_module, target_layer_class)
        else:
            is_target = type(sub_module) is target_layer_class

        if is_target:
            target_submodules.append(sub_module)
        elif len(list(sub_module.children())) > 0:
            target_submodules += get_all_submodules(
                sub_module, target_layer_class, include_subclasses)

    return target_submodules


def wrap_hf_module(module, wrapper_layer_class, target_layer_class,
                   **wrapper_layer_kwargs):
    '''Wrap all layers in module of type target_layer_class with 
       wrapper_layer_class (either a class or a hyhdra config file)'''

    for name, sub_module in reversed(module._modules.items()):
        if type(sub_module) is target_layer_class:
            print('SWAPPING')

            wrapper_layer_kwargs['config'] = sub_module.config
            # model_config = sub_module.config
            if hasattr(sub_module, 'layer_idx'):
                wrapper_layer_kwargs['layer_idx'] = sub_module.layer_idx
            if isinstance(wrapper_layer_class, DictConfig):
                wrapped_module = hydra.utils.instantiate(
                    **wrapper_layer_kwargs)
            else:
                wrapped_module = wrapper_layer_class(
                    # model_config=model_config,
                    **wrapper_layer_kwargs)
            module._modules[name] = wrapped_module
        elif len(list(sub_module.children())) > 0:

            module._modules[name] = wrap_hf_module(
                sub_module, wrapper_layer_class, target_layer_class,
                **wrapper_layer_kwargs)

    return module


def compute_masked_statistics(values, mask, reduce_dims,):
    '''Computing sample mean, summed variances, and number of elements, keeping 
       reduction dimensions to 1'''
    mask = mask.expand_as(values)

    masked_values = torch.where(mask, values, torch.zeros_like(values))

    total_num = mask.to(dtype=torch.long).sum(dim=reduce_dims, keepdim=True)
    mean = masked_values.sum(dim=reduce_dims, keepdim=True)/total_num
    variance_sum = (masked_values - mean).square().sum(
        dim=reduce_dims, keepdim=True)
    return mean, variance_sum, total_num


def merge_statistics(mean_a, variance_sum_a, num_a,
                     mean_b, variance_sum_b, num_b,):
    '''Computes statistics of whole population using the stable algorithm from
       http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf'''

    num_ab = num_a + num_b
    mean_diff = mean_b - mean_a
    a_coeff = num_a/num_ab
    b_coeff = 1 - a_coeff
    mean_ab = a_coeff*mean_a + b_coeff*mean_b
    variance_sum_ab = (variance_sum_a + variance_sum_b +
                       mean_diff.square()*(a_coeff*num_b))
    return mean_ab, variance_sum_ab, num_ab


def compute_masked_statistics_with_var(values, mask, reduce_dims,):
    '''Computing sample mean, variances, and number of elements, keeping 
       reduction dimensions to 1'''
    mask = mask.expand_as(values)

    masked_values = torch.where(mask, values, torch.zeros_like(values))

    total_num = mask.to(dtype=torch.long).sum(dim=reduce_dims, keepdim=True)

    clamped_num = torch.clamp_min(total_num, min=1)

    mean = masked_values.sum(dim=reduce_dims, keepdim=True)/clamped_num

    # raise NotImplementedError
    diffs = masked_values - mean
    masked_diffs = torch.where(mask, diffs, torch.zeros_like(diffs))
    variance = masked_diffs.square().sum(
        dim=reduce_dims, keepdim=True)/clamped_num
    return mean, variance, total_num


def merge_statistics_from_var(mean_a, variance_a, num_a,
                              mean_b, variance_b, num_b,):
    '''Computes statistics of whole population directly from variance estimates,
       adapting the algorithm in:
       http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
       to avoid floating point overflow'''

    num_ab = num_a + num_b
    # need to keep as original dtype to avoid overflows (cannot add a small eps)
    clamped_num_ab = torch.clamp_min(num_ab, min=1)
    mean_diff = mean_b - mean_a
    a_coeff = num_a/clamped_num_ab
    b_coeff = 1 - a_coeff

    # raise NotImplementedError
    mean_ab = a_coeff*mean_a + b_coeff*mean_b
    variance_ab = (a_coeff*variance_a + b_coeff*variance_b +
                   mean_diff.square()*(a_coeff*b_coeff))

    return mean_ab, variance_ab, num_ab


def faster_attn_reversecumsum(tensor, dim=-1, **kwargs):
    '''Faster reverse cumsum operation, relying on the fact that the attention 
       matrix will sum to 1. Due to roundoffs in precision, it might lead to 
       small numerical differences w/ .flip().cumsum()'''

    # each term will subtract all previous terms from 1
    reverse_sum = 1 + tensor - torch.cumsum(tensor, dim=dim, **kwargs)
    return reverse_sum


def pad_and_stack_attn_mxs(
        unpacked_attn_mx: List[torch.Tensor], move_to_gpu=True,
        lens=None, max_len=None, padding_side='left', return_lens=False):
    if lens is None:
        lens = [t.size(-1) for t in unpacked_attn_mx]
    if max_len is None:
        max_len = max(lens)
    packed_attn_mx = []
    for l, t in zip(lens, unpacked_attn_mx):
        pad_amount = max_len - l
        if padding_side == 'left':
            pad_tuple = [pad_amount, 0, pad_amount, 0]
        else:
            pad_tuple = [0, pad_amount, 0, pad_amount]
        padded_t = F.pad(t, pad=pad_tuple)
        packed_attn_mx.append(padded_t)
    packed_attn_mx = torch.stack(packed_attn_mx, dim=0)
    if move_to_gpu:
        packed_attn_mx = packed_attn_mx.cuda()
    if return_lens:
        return packed_attn_mx, lens, max_len
    return packed_attn_mx


def pad_and_concat_buffered_attn_mxs(
        # list of buffered attn mx from oldest to newest
        buffered_attn_mxs: List[torch.Tensor],
        move_to_gpu=False,
        # right for causal attn
        padding_side='right',):
    '''Pads and concatenates attention mxs into a single one for all the final 
       tokens'''
    final_all_tokens = buffered_attn_mxs[-1].shape[-1]
    # padded_attn_mxs = []
    for i, attn_mx in enumerate(buffered_attn_mxs[:-1]):
        pad_amount = final_all_tokens - attn_mx.shape[-1]
        if padding_side == 'left':
            pad_tuple = [pad_amount, 0]
        else:
            pad_tuple = [0, pad_amount]
        # padded_attn_mx = F.pad(attn_mx, pad=pad_tuple)
        # update deleting old data to free memory
        buffered_attn_mxs[i] = F.pad(attn_mx, pad=pad_tuple, value=0)
    if move_to_gpu:
        return torch.concat(buffered_attn_mxs, dim=-2).cuda()
    return torch.concat(buffered_attn_mxs, dim=-2)


def pack_attn_mxs(
        # num_samples x num_layers # of mxs
        unpacked_attn_mxs: List[List[torch.Tensor]], move_to_gpu=True,
        padding_side='left'):
    # num_layers x num_samples # of mxs
    attn_mxs_per_layer = list(zip(*unpacked_attn_mxs))

    # avoid recomputing, should be the same for all layers
    lens = None
    max_len = None

    packed_mxs = []
    for attn_mxs in attn_mxs_per_layer:
        stacked_mxs, lens, max_len = pad_and_stack_attn_mxs(
            unpacked_attn_mx=attn_mxs, move_to_gpu=move_to_gpu, lens=lens,
            max_len=max_len, padding_side=padding_side, return_lens=True
        )
        packed_mxs.append(stacked_mxs)
    return packed_mxs


def split_attn_mx_from_attn_mask(
        mx: torch.Tensor, attn_mask: torch.Tensor, move_to_cpu: bool = True,
        unpack_dim: int = 0) -> torch.Tensor:
    # attn mask used for LAST dimension
    if move_to_cpu:
        mx = mx.detach().cpu()
    unpacked_out = []
    for i, mx_i in enumerate(mx.unbind(dim=unpack_dim)):
        mask_i = attn_mask[i].long()
        num_unmasked = torch.sum(mask_i)
        if num_unmasked < mask_i.size(0):
            if mask_i[0] == 0:  # left side padding
                unpacked_out.append(mx_i[..., -num_unmasked:, -num_unmasked:])
            elif mask_i[-1] == 0:  # right side padding
                unpacked_out.append(mx_i[..., :num_unmasked, :num_unmasked])
            else:
                print('Error: masked tokens not due to padding')
                raise NotImplementedError
        else:
            unpacked_out.append(mx_i)
    return unpacked_out


def unpack_attn_mxs_from_attn_mask(
        mxs: List[torch.Tensor], attn_mask: torch.Tensor,
        move_to_cpu: bool = True, unpack_dim: int = 0) -> List[torch.Tensor]:
    # attn mask used for LAST dimension
    unpacked_mxs_per_layer = []
    for layer_i, mx in enumerate(mxs):
        unpacked_mx = split_attn_mx_from_attn_mask(
            mx=mx, attn_mask=attn_mask, move_to_cpu=move_to_cpu,
            unpack_dim=unpack_dim)
        unpacked_mxs_per_layer.append(unpacked_mx)
    return list(zip(*unpacked_mxs_per_layer))


def unpack_kv_cache(kv_cache: list, attn_mask: torch.Tensor,
                    move_to_cpu: bool = True):

    num_layers = len(kv_cache)  # should be num. of layers
    num_tensors_per_l = len(kv_cache[0])  # should be 2
    num_samples = kv_cache[0][0].size(0)

    # initial unpacked cache of None
    unpacked_cache = []
    for _ in range(num_samples):
        unpacked_sample_cache = []
        for _ in range(num_layers):
            temp_tensor_cache = [None for _ in range(num_tensors_per_l)]
            unpacked_sample_cache.append(temp_tensor_cache)
        unpacked_cache.append(unpacked_sample_cache)

    # mask_per_sample = torch.unbind(attn_mask, dim=0)

    # iterate through layers
    for i, layer_tensors in enumerate(kv_cache):
        # iterate through kv
        for j, tensor in enumerate(layer_tensors):
            if move_to_cpu:
                tensor = tensor.detach().cpu()
            tensors_per_sample = torch.unbind(tensor, dim=0)

            for k, unpacked_tensor in enumerate(tensors_per_sample):
                mask_for_k = attn_mask[k]
                mask_for_k = mask_for_k.to(device=tensor.device)
                unpacked_cache[k][i][j] = unpacked_tensor[...,
                                                          mask_for_k.bool(), :]

    return unpacked_cache


def concat_and_pad(input_list, lens=None, max_len=None, return_mask=True,
                   padding_side='left'):
    # concatenating using dimension -2
    if lens is None:
        lens = [t.size(0) for t in input_list]
    if max_len is None:
        max_len = max(lens)

    padded_inputs = []
    rec_attn_masks = []

    for l, t in zip(lens, input_list):
        pad_amount = max_len - l
        pad_values = torch.zeros(
            [pad_amount], device=t.device, dtype=torch.long)
        if return_mask:
            mask_ones = torch.ones([l], device=t.device, dtype=torch.long)
        # NOTE: add unsqueeze + expand for kv cache
        if padding_side == 'left':
            pad_tuple = [0, 0, pad_amount, 0]
        else:
            pad_tuple = [0, 0, 0, pad_amount]
        padded_input_i = F.pad(t, pad=pad_tuple)
        if return_mask:
            # each mask vector is 1d
            mask_i = F.pad(mask_ones, pad=pad_tuple[-2:])
        padded_inputs.append(padded_input_i)
        if return_mask:
            rec_attn_masks.append(mask_i)

    padded_inputs = torch.stack(padded_inputs, dim=0)
    if return_mask:
        rec_attn_masks = torch.stack(rec_attn_masks, dim=0)
    return padded_inputs, rec_attn_masks, lens, max_len


def pack_kv_cache(unpacked_cache, move_to_gpu=True, padding_side='left'):
    num_samples = len(unpacked_cache)
    num_tensor_lists = len(unpacked_cache[0])
    num_tensors = len(unpacked_cache[0][0])

    lens = [unpacked_cache[i][0][0].size(-2) for i in range(num_samples)]
    max_len = max(lens)

    attn_mask = None

    packed_kv_cache = []
    for i in range(num_tensor_lists):
        tensor_list = []
        for j in range(num_tensors):
            packed_tensor_cache_list = [unpacked_cache[k][i][j]
                                        for k in range(num_samples)]
            compute_attn_mask = attn_mask is None
            outputs = concat_and_pad(
                packed_tensor_cache_list, lens=lens, max_len=max_len,
                return_mask=compute_attn_mask, padding_side=padding_side)
            packed_tensor = outputs[0]
            if compute_attn_mask:
                attn_mask = outputs[1]
            if move_to_gpu:
                packed_tensor = packed_tensor.cuda()
            tensor_list.append(packed_tensor)
        packed_kv_cache.append(tensor_list)

    return packed_kv_cache, attn_mask


def load_results_file(results_path):
    with open(results_path, 'r') as f:
        loaded_res = json.load(f)
        if isinstance(loaded_res, list):
            loaded_res = loaded_res[0]
    return loaded_res


def aggregate_score_dict(
        score_dict,
        score_aggregation='mean',
        score_normalization_reference=None,
        sample_idxs_per_task=None,
        task_names=None,
):
    # exclude 'statistics' entries (in dict objects)
    if task_names is not None:
        norm_score_dict = {
            task_n: score for task_n, score in score_dict.items()
            if not isinstance(score, dict) and task_n in task_names}
    else:
        norm_score_dict = {task_n: score for task_n, score in score_dict.items()
                           if not isinstance(score, dict)}
    if score_normalization_reference is None:
        for task_n, score in norm_score_dict.items():
            if task_n.startswith('lb/'):
                norm_score_dict[task_n] = score/100
    else:
        for task_n, score in norm_score_dict.items():
            normalizer_scores = score_normalization_reference[task_n]
            if sample_idxs_per_task is not None:
                filt_normalizer_scores = [normalizer_scores[str(i)] for i in
                                          sample_idxs_per_task[task_n]]
            else:
                filt_normalizer_scores = list(normalizer_scores.values())
            normalizer = np.mean(filt_normalizer_scores)
            if task_n.startswith('lb/'):
                normalizer *= 100
            norm_score_dict[task_n] = score/(normalizer + 1e-7)
    if score_aggregation == 'mean':
        score = np.mean(list(norm_score_dict.values()))
    else:
        raise NotImplementedError
    if score_normalization_reference is None:
        score = score/100
    return score, norm_score_dict


def zip_dict(dict_of_dicts, inner_keys=None):
    # unpack and zips dictionary of dictionary (assumed to be sharing the same
    # keys if not specified) reversing the inner-outer structure
    if inner_keys is None:
        first_inner_dict = list(dict_of_dicts.values())[0]
        inner_keys = list(first_inner_dict.keys())
    reversed_dict_of_dicts = {
        inner_k: {outer_k: dict_of_dicts[outer_k][inner_k]
                  for outer_k in dict_of_dicts} for inner_k in inner_keys}
    return reversed_dict_of_dicts


def merge_dicts(dict1, dict2, suffix, keys=None):
    if keys is None:
        keys = dict1.keys()
    merged_dict = copy.deepcopy(dict1)
    merged_dict.update({f'{k}{suffix}': copy.deepcopy(v)
                        for k, v in dict2.items()})
    return merged_dict


class CtxCollator(Collator):
    # extended from https://github.com/EleutherAI/lm-evaluation-harness
    @staticmethod
    def get_chunks(_iter, n: int = 0, fn=None,):
        arr = []
        _iter = tuple(_iter)
        ctx_len = None
        for i, x in enumerate(_iter):
            arr.append(x)
            if fn:
                target_len, ctx_len = fn(i, _iter)
            else:
                target_len = n
            if len(arr) == target_len:
                yield arr, ctx_len
                arr = []

        if arr:
            yield arr, ctx_len

    def get_batched(self, n: int = 1, batch_fn=None, reorder=True):
        if self._group_by == "gen_kwargs":
            for (
                key,
                values,
            ) in self._arr_with_indices.items():  # type: ignore
                values = self._reorder(values, reorder=reorder)
                batch = self.get_chunks(values, n=n, fn=batch_fn)
                yield from batch

        elif self._group_by == "contexts":
            # Get one sample from each key
            values = self._reorder(
                [value[0] for value in self._arr_with_indices.values()]
            )

            batch = self.get_chunks(values, n=n, fn=batch_fn)
            yield from batch
        else:
            values = self._reorder(self._arr_with_indices)  # type: ignore
            batch = self.get_chunks(values, n=n, fn=batch_fn)
            yield from batch

    def _reorder(self, arr, reorder=True):
        if reorder:
            arr = sorted(arr, key=self._sort_fn)
        if not self._group_by == "contexts":
            # If grouped by contexts then indices will be set in get_cache()
            self._reorder_indices.extend([x[0] for x in arr])
        yield from [x[1] for x in arr]


def load_hf_model(model_name: str, model_kwargs={}, tokenizer_kwargs={}):
    tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, **tokenizer_kwargs)
    return model, tokenizer


def convert_to_dict_of_lists(result_dicts: dict):
    aggregated_result_dict = {}
    for result_dict in result_dicts:
        for k, v in result_dict.items():
            if isinstance(v, dict):
                continue
            current_aggregated_result = aggregated_result_dict.get(k, [])
            current_aggregated_result.append(v)
            aggregated_result_dict[k] = current_aggregated_result
    return aggregated_result_dict


def concat_list_of_dicts_of_lists(result_dicts_list: List[dict]):
    aggregated_result_dict = {}
    result_dict_0 = result_dicts_list[0]
    for k, v in result_dict_0.items():
        if isinstance(v, dict):
            continue
        current_aggregated_result = aggregated_result_dict.get(k, [])
        current_aggregated_result.append(v)
        aggregated_result_dict[k] = np.concatenate(
            [c_dict[k] for c_dict in result_dicts_list], axis=0)
    return aggregated_result_dict


def pop_stats_from_dict_of_lists(aggregated_result_dict: dict, prefix=None):
    stats_dict = {}
    if prefix is None:
        prefix = ''
    for k, l in aggregated_result_dict.items():
        stats_dict[prefix + k + '_mean'] = np.mean(l)
        stats_dict[prefix + k + '_std'] = np.std(l)
        stats_dict[prefix + k + '_best'] = np.max(l)
        stats_dict[prefix + k + '_worst'] = np.min(l)
    return stats_dict


class COLOR:
    # ANSI color codes and tools
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"
