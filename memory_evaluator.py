import copy
import os
import traceback
from typing import List, Optional, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from accelerate import (
    find_executable_batch_size,
)

from utils import (
    empty_gpu_cache, pack_kv_cache, pack_attn_mxs, is_oom_exception,
    get_first_value_split)


from tqdm import tqdm

# from lm_eval.api.model import TemplateLM
from lm_eval.models.utils import (
    clear_torch_cache,
    stop_sequences_criteria,
)

from utils import CtxCollator

import json

from utils_longbench import build_chat


class MemoryHFEvaluator():
    def __init__(
        self,
        model,
        tokenizer,
        evaluation_ctx_steps=1,
        add_bos_token=False,  # add beginning of sentence token
        eval_max_batch_size=128,  # 8192, # 32 x 256

        memory_batch_size=False,  # number of memories to evaluate
        batch_size=None,
        max_conditioning_length=None,  # 4096,
        max_memory_length=2048,
        max_gen_tokens=512,  # from the HFLM class
        # whether to start using the KV cache only for newly generated tokens
        full_context_gen=True,
        # Option to save memory by recomputing the softmax and concatenating
        # samples at each timestep individually (saves memory at the cost of
        # some speed). Setting to False, might invalidate the automatic batch
        # size inference
        per_timestep_loglikelihood: bool = True,
        # should help alleviate pt memory issues in interactive sessions
        force_clear_cache=True,
        device=None,

        # error handling
        max_retry_iter: int = 5,
        log_misc: bool = False,
    ) -> None:
        super().__init__()

        self.log_misc = log_misc
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.setup_tokenizer_padding()
        else:
            print('WARNING: tokenizer is none, evalluator will not be usable')

        self.device = device or self.model.device
        self.config = self.model.config
        self.evaluation_ctx_steps = evaluation_ctx_steps
        self.add_bos_token = add_bos_token
        self.max_batch_size = eval_max_batch_size
        self.batch_size = batch_size
        if batch_size is None:
            self.batch_size = eval_max_batch_size
        self.memory_batch_size = memory_batch_size
        self.max_conditioning_length = max_conditioning_length
        self.max_memory_length = max_memory_length

        if hasattr(model, 'memory_policy'):
            if self.model.memory_policy.cache_size is not None:
                self.max_memory_length = min(
                    self.max_memory_length, self.model.memory_policy.cache_size)
            self.memory_policy = self.model.memory_policy
            self.is_memory_model = True
        else:
            self.is_memory_model = False

        self.dynamic_scaling = False
        if self.max_conditioning_length is None:
            config = model.config
            if hasattr(config, 'max_position_embeddings'):
                self.max_conditioning_length = config.max_position_embeddings
                if hasattr(config, 'rope_scaling'):
                    if config.rope_scaling is not None:
                        if 'dynamic' in config.rope_scaling['type']:
                            self.max_conditioning_length = None
                            self.dynamic_scaling = True
                        else:
                            self.max_conditioning_length *= (
                                config.rope_scaling['factor'])
            else:
                # use model2maxlen specified in longbench
                model2maxlen = json.load(open(
                    "LongBench/config/model2maxlen.json", "r"))

                model_suffix = self.model_name.split('/')[-1]
                if model_suffix in model2maxlen:
                    max_length = model2maxlen[model_suffix]
                    self.max_conditioning_length = max_length

            if (self.max_conditioning_length is None and
                    not self.dynamic_scaling):
                raise ValueError("Could not infer max_conditioning length for" +
                                 "the input model, please pass in manually")

        self.batch_schedule = 1
        self.ctx_schedule = 1
        self.batch_sizes = {}
        self.ctx_sizes = {}
        self.max_gen_toks = max_gen_tokens
        self.full_context_gen = full_context_gen

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(
                batch_size) > 1 else 1
            self.ctx_schedule = 1
        else:
            self.batch_size_per_gpu = int(batch_size)
            self.ctx_schedule = 1

        self.per_timestep_loglikelihood = per_timestep_loglikelihood
        self.force_clear_cache = force_clear_cache
        self.max_retry_iter = max_retry_iter

    @property
    def model_name(self,):
        if hasattr(self.model, 'model_name'):
            return self.model.model_name
        else:
            return self.model.config.name_or_path

    def swap_memory_policy(self, new_memory_policy):
        self.model.swap_memory_policy(new_memory_policy=new_memory_policy)
        self.memory_policy = new_memory_policy
        if self.model.memory_policy.cache_size is not None:
            self.max_memory_length = min(
                self.max_memory_length, self.model.memory_policy.cache_size)
        self.memory_policy.cuda()

    def setup_tokenizer_padding(self,):
        # taken from huggingface lm_eval code for consistency
        if self.tokenizer.pad_token:
            pass
        elif self.tokenizer.unk_token:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        elif self.tokenizer.eos_token:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            if getattr(self.config, "model_type", None) == "qwen":
                self.tokenizer.pad_token = "<|endoftext|>"
            elif (
                self.tokenizer.__class__.__name__ == "RWKVWorldTokenizer"
                or self.tokenizer.__class__.__name__ == "Rwkv5Tokenizer"
            ):
                assert self.tokenizer.pad_token_id == 0
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    def _detect_ctx_size(self, batch_size, requests=None, pos: int = 0):
        max_memory_length = self.max_memory_length
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_requests_len = len(
                (context_enc + continuation_enc)[-(
                    self.max_conditioning_length + 1):][:-1]
            )
            max_memory_length = min(max_memory_length, max_requests_len)
        empty_gpu_cache()

        @find_executable_batch_size(starting_batch_size=max_memory_length)
        def forward_ctx(ctx_size):
            call_kwargs = {}
            test_batch = torch.ones(
                (batch_size, ctx_size), device=self.device
            ).long()
            past_key_values = None
            if self.is_memory_model:
                param_idxs = np.zeros([batch_size], dtype=int)
                self.memory_policy.set_params_batch_idxs(param_idxs=param_idxs)
            num_iters = max_memory_length // ctx_size
            for i in range(num_iters):
                logits, past_key_values = self._model_call(
                    test_batch, past_key_values=past_key_values, use_cache=True,
                    **call_kwargs)
                out = F.log_softmax(logits, dim=-1)
            return ctx_size

        try:
            ctx_size = forward_ctx()
        except RuntimeError as e:
            if "No executable ctx size found" in str(e):
                ctx_size = 1
            else:
                raise

        empty_gpu_cache()

        if self.world_size > 1:
            # if multi-GPU, always take minimum over all selected batch sizes
            max_rnk_bs = torch.tensor([batch_size], device=self.device)
            gathered = (
                self.accelerator.gather(
                    max_rnk_bs).cpu().detach().numpy().tolist()
            )
            batch_size = min(gathered)
            # same for ctx size
            max_rnk_ctx = torch.tensor([ctx_size], device=self.device)
            gathered = (
                self.accelerator.gather(
                    max_rnk_ctx).cpu().detach().numpy().tolist()
            )
            ctx_size = min(gathered)
            clear_torch_cache()
            return batch_size, ctx_size

        clear_torch_cache()
        return ctx_size

    def _detect_batch_size(self, requests=None, pos: int = 0):
        max_memory_length = self.max_memory_length
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_requests_len = len(
                (context_enc +
                 continuation_enc)[-(self.max_conditioning_length + 1):][:-1]
            )
            max_memory_length = min(max_memory_length, max_requests_len)

        empty_gpu_cache()

        # if OOM, then halves batch_size and tries again
        @find_executable_batch_size(starting_batch_size=self.max_batch_size)
        def forward_batch(batch_size):
            call_kwargs = {}
            test_batch = torch.ones(
                (batch_size, max_memory_length//5), device=self.device
            ).long()
            past_key_values = None
            if self.is_memory_model:
                param_idxs = np.zeros([batch_size], dtype=int)
                self.memory_policy.set_params_batch_idxs(param_idxs=param_idxs)
            for i in range(5):  # two loops should be enough
                logits, past_key_values = self._model_call(
                    test_batch, past_key_values=past_key_values, use_cache=True,
                    **call_kwargs)
                out = F.log_softmax(logits, dim=-1)
            return batch_size

        try:
            batch_size = forward_batch()
        except RuntimeError as e:
            if "No executable batch size found" in str(e):
                batch_size = 1
            else:
                raise

        # Find max first ctx length
        @find_executable_batch_size(starting_batch_size=max_memory_length)
        def forward_ctx(ctx_size):
            call_kwargs = {}
            test_batch = torch.ones(
                (batch_size, ctx_size), device=self.device
            ).long()
            past_key_values = None
            if self.is_memory_model:
                param_idxs = np.zeros([batch_size], dtype=int)
                self.memory_policy.set_params_batch_idxs(param_idxs=param_idxs)
            num_iters = max_memory_length // ctx_size
            for i in range(num_iters):
                logits, past_key_values = self._model_call(
                    test_batch, past_key_values=past_key_values, use_cache=True,
                    **call_kwargs)
                out = F.log_softmax(logits, dim=-1)  # noqa: F841
            return ctx_size

        try:
            ctx_size = forward_ctx()
        except RuntimeError as e:
            if "No executable ctx size found" in str(e):
                ctx_size = 1
            else:
                raise

        empty_gpu_cache()

        if self.world_size > 1:
            # if multi-GPU, always take minimum over all selected batch sizes
            max_rnk_bs = torch.tensor([batch_size], device=self.device)
            gathered = (
                self.accelerator.gather(max_rnk_bs).cpu().detach().numpy(
                ).tolist()
            )
            batch_size = min(gathered)
            # same for ctx size
            max_rnk_ctx = torch.tensor([ctx_size], device=self.device)
            gathered = (
                self.accelerator.gather(max_rnk_ctx).cpu().detach().numpy(
                ).tolist()
            )
            ctx_size = min(gathered)
            clear_torch_cache()
            return batch_size, ctx_size

        clear_torch_cache()
        return batch_size, ctx_size

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:

        special_tokens_kwargs = {}
        if add_special_tokens is None:
            special_tokens_kwargs = {"add_special_tokens": False or
                                     self.add_bos_token}
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer(string, **special_tokens_kwargs).input_ids
        # left-truncate the encoded context to be at most left_truncate_len long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
        use_mid_cropping: bool = False,
        build_chat_interface: bool = False,
        max_conditioning_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if max_conditioning_length is None:
            max_conditioning_length = self.max_conditioning_length
        # encode a batch of strings. converts to tensors and pads automatically,
        # unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {
            "add_special_tokens": False or self.add_bos_token}

        # pads to longest seq. in batch, if truncation:
        if use_mid_cropping and max_conditioning_length is not None:
            new_strings = []
            for s in strings:
                s_input_ids = self.tokenizer(
                    s,
                    truncation=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids[0]

                encoding_len = len(s_input_ids)

                if max_conditioning_length < encoding_len:
                    half = max_conditioning_length // 2
                    new_string = (self.tokenizer.decode(
                        s_input_ids[:half], skip_special_tokens=True) +
                        self.tokenizer.decode(
                            s_input_ids[-half:], skip_special_tokens=True))
                    new_strings.append(new_string)
                else:
                    new_strings.append(s)
            strings = new_strings

        if build_chat_interface:
            tokenized_input_ids = []
            masks = []
            for s in strings:
                tokenized_s = build_chat(s)
                tokenized_input_ids.append(tokenized_s.input_ids[0])
                masks.append(tokenized_s.input_mask[0])
                raise NotImplementedError

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if truncation and (left_truncate_len is not None):
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side
        return encoding["input_ids"], encoding["attention_mask"]

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(
            tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps, past_key_values, attention_mask=None,
                    use_cache=False):
        model_out = self.model(inps, past_key_values=past_key_values,
                               attention_mask=attention_mask,
                               use_cache=use_cache,)
        return model_out.logits, model_out.past_key_values

    def _encode_and_generate(self,
                             contexts,
                             shared_tok_kwargs,
                             shared_gen_kwargs,
                             chunk_pop_idxs=None,
                             chunk_precached_tensors=None,
                             retry_iter=0,
                             # this option skips generation
                             # used to record the datasets statistics.
                             return_encoding_size=False,
                             ):
        bs = len(contexts)
        if retry_iter > self.max_retry_iter:
            raise ValueError('Unable to find batch size not causing OOM. ' +
                             'Maximum specified retries attempts exceeded.')
        elif retry_iter == 1 and bs == 1:
            print('Since batch_size = 1, retrying once without decreasing it')
            if self.is_memory_model:
                if self.model.max_new_tokens is not None:
                    if self.model.max_new_tokens % 2 == 0:
                        limit_new_tokens = self.model.max_new_tokens // 2
                        shared_gen_kwargs["limit_new_tokens"] = limit_new_tokens
                        print('Instead, limiting new tokens to ' +
                              f'{limit_new_tokens}')
            current_splits = 1
            new_max_bs = bs
        else:
            current_splits = 2**retry_iter
            new_max_bs = bs//current_splits
            if new_max_bs == 0:
                raise ValueError('Unable to find batch size not causing OOM. ' +
                                 'Even batch_size=1 causes OOM during gen.')

        new_batches_split_idxs = list(range(0, bs, new_max_bs)) + [bs]

        if self.log_misc:
            print(f'Retry {retry_iter} split idxs {new_batches_split_idxs}')

        if "max_length" not in shared_gen_kwargs:
            if "max_new_tokens" not in shared_gen_kwargs:
                shared_gen_kwargs["max_new_tokens"] = self.max_gen_toks

        cont_list = []
        for i, start_batch_idx in enumerate(new_batches_split_idxs[:-1]):
            gen_kwargs = copy.deepcopy(shared_gen_kwargs)
            end_batch_idx = new_batches_split_idxs[i+1]
            curr_contexts = contexts[start_batch_idx:end_batch_idx]
            if chunk_pop_idxs is not None:
                curr_pop_idxs = chunk_pop_idxs[start_batch_idx:end_batch_idx]
                self.memory_policy.set_params_batch_idxs(
                    param_idxs=curr_pop_idxs)

            context_enc, attn_masks = self.tok_batch_encode(
                curr_contexts,
                **shared_tok_kwargs,
            )
            context_enc = context_enc.to(self.device)
            attn_masks = attn_masks.to(self.device)

            if return_encoding_size:
                cont_list.append(context_enc.shape[-1])
                continue

            if self.log_misc:
                print(f'CTX - LEN {context_enc.shape}')

            if chunk_precached_tensors is not None:
                curr_precached_tensors = chunk_precached_tensors[
                    start_batch_idx:end_batch_idx]
                unpacked_kv_cache, unpacked_attn_mxs = list(zip(
                    *curr_precached_tensors))

                packed_kv_cache, relative_attn_mask = pack_kv_cache(
                    unpacked_cache=unpacked_kv_cache, move_to_gpu=True,
                    padding_side='left',)
                packed_attn_mx = pack_attn_mxs(
                    unpacked_attn_mxs=unpacked_attn_mxs, move_to_gpu=True,
                    padding_side='left',)
                if self.is_memory_model:
                    self.model.load_cached_attn_mxs(
                        cached_attn_mxs=packed_attn_mx)

                gen_kwargs['past_key_values'] = packed_kv_cache
                context_enc = context_enc[..., -1:]
                raise NotImplementedError

            generation_out = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                **gen_kwargs,
            )

            cont = generation_out[:, context_enc.shape[1]:]
            if self.log_misc:
                print(f'OUTPUT CONTENT - LEN {cont.shape}')
            cont_list += cont.tolist()
        return cont_list

    def _model_generate(self, context, stop,
                        tok_stop=False,
                        **generation_kwargs):
        generation_kwargs["temperature"] = generation_kwargs.get(
            "temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # temp == 0 gets converted do_sample = false
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False
        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")

        # build stopping criteria
        if tok_stop:
            stop_kwargs = dict(eos_token_id=stop)
        else:
            stopping_criteria = stop_sequences_criteria(
                tokenizer=self.tokenizer,
                stop_sequences=stop,
                initial_decoder_input_length=context.shape[1],
                batch_size=context.shape[0]
            )
            stop_kwargs = dict(stopping_criteria=stopping_criteria)

        if self.is_memory_model:
            # for now, the attention mask should be always specified
            # thus, serving as a check
            attn_mask: torch.Tensor = generation_kwargs['attention_mask']
            max_seq_lens = attn_mask.sum(-1, keepdim=True)
            max_seq_lens_after_gen = max_seq_lens + self.max_gen_toks
            self.model.set_max_seq_lens(max_seq_lens_after_gen)

        # NOTE: since use_cache = True, this should work with simple wrapper
        # around attn layers
        if self.full_context_gen or not self.is_memory_model:

            generated_outputs = self.model.generate(
                input_ids=context,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                **stop_kwargs,
                **generation_kwargs,
            )
        else:
            raise NotImplementedError
        if self.is_memory_model:
            self.model.set_max_seq_lens(None)

        return generated_outputs

    def _select_cont_toks(
        self,
        logits: torch.Tensor,
        relevant_content_length: int = None,
        full_ctx_length: int = None

    ) -> torch.Tensor:
        logits = logits[full_ctx_length -
                        relevant_content_length:full_ctx_length]
        return logits

    def _batch_scheduler(self, pos, n_reordered_requests):
        sched = pos // int(len(n_reordered_requests) / self.batch_schedule)
        if sched in self.batch_sizes:
            return self.batch_sizes[sched], self.ctx_sizes[sched]
        if (len(self.batch_sizes) > 1) and (
            self.batch_sizes[sched - 1] == self.max_batch_size
        ):
            # if previous batch size is already maximal, skip recomputation
            self.batch_sizes[sched] = self.max_batch_size
            return self.batch_sizes[sched], self.ctx_sizes[sched]
        print(
            f"Passed argument batch_size = auto:{self.batch_schedule}. " +
            "Detecting largest batch size"
        )
        self.batch_sizes[sched], self.ctx_sizes[sched] = (
            self._detect_batch_size(n_reordered_requests, pos))
        print(f"Determined largest batch size: {self.batch_sizes[sched]}")
        return self.batch_sizes[sched], self.ctx_sizes[sched]

    def evaluate_lb(
        self,
        # dataset_name: str,
        dataset_samples: List[str],
        # build_chat_prompts: bool = False,
        disable_tqdm: bool = False,
        pop_reps: int = 1,
        pop_idxs: Optional[np.array] = None,
        # if precaching = True, only computes the initial kv_cache to store
        precaching: bool = False,
        precached_tensors: bool = None,
        max_gen_tokens: Optional[int] = None,
        stop_gen: Optional[List[str]] = [],
        build_chat_interface: bool = False,
        model_kwargs: dict = {},
    ) -> List[str]:

        res = []
        using_precache = precached_tensors is not None

        if max_gen_tokens is None:
            max_gen_tokens = self.max_gen_toks

        tok_stop_gen = [self.eot_token_id]

        if using_precache:
            assert not precaching

        def _collate(req: str):
            """Defines the key for the sorted method"""
            toks = self.tok_encode(req)
            return -len(toks), req

        if (pop_reps > 1) or (pop_idxs is not None):
            assert self.is_memory_model, ('Not using a memory model but passing'
                                          ' an increased population value.')
            if pop_idxs is None:
                pop_idxs = np.repeat(np.arange(pop_reps), len(dataset_samples))
            else:
                assert len(pop_idxs) == pop_reps, (
                    'pop_idxs should be of size # of pop_reps')

                pop_idxs = np.repeat(pop_idxs, len(dataset_samples))

            dataset_samples = dataset_samples*pop_reps  # repeat pop_reps times
            if using_precache:
                precached_tensors = precached_tensors*pop_reps
        elif self.is_memory_model and pop_idxs is None:
            pop_idxs = np.zeros([len(dataset_samples)], dtype=int)

        if self.log_misc:
            pbar = tqdm(
                total=len(dataset_samples),
                disable=(disable_tqdm or (int(os.getenv("RANK")) != 0)),
                desc="Running longbench requests",
            )
        adaptive_batch_size = None

        if self.batch_size == "auto":
            # using rolling window with maximum context
            print('Passed argument batch_size = auto. Detecting largest batch' +
                  'size')
            batch_size, ctx_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size
            if adaptive_batch_size is not None
            else 0
        )

        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and not adaptive_batch_size
            else None
        )

        re_ords = CtxCollator(
            dataset_samples,
            sort_fn=_collate,
            group_by=None,
            group_fn=lambda x: x[1],
        )

        chunks = list(re_ords.get_batched(
            n=batch_size, batch_fn=batch_fn, reorder=True,))

        # set the max length in tokens of inputs ("context_enc")
        if self.max_conditioning_length is not None:
            max_prompt_conditioning = (self.max_conditioning_length -
                                       max_gen_tokens)
        else:
            max_prompt_conditioning = None
        max_ctx_len = max_prompt_conditioning

        shared_tok_kwargs = dict(
            padding_side="left",
            left_truncate_len=max_ctx_len,
            truncation=False,
            use_mid_cropping=True,
            build_chat_interface=build_chat_interface,
            max_conditioning_length=max_ctx_len,
        )

        shared_gen_kwargs = dict(
            stop=tok_stop_gen,
            tok_stop=True,
            max_new_tokens=max_gen_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )

        shared_gen_kwargs.update(model_kwargs)

        processed_reqs = 0

        for i, (contexts, _) in enumerate(chunks):
            start_idx = processed_reqs
            end_idx = start_idx + len(contexts)
            chunk_indices = re_ords._reorder_indices[start_idx:end_idx]

            if pop_idxs is not None:
                chunk_pop_idxs = pop_idxs[chunk_indices]
                self.memory_policy.set_params_batch_idxs(
                    param_idxs=chunk_pop_idxs)
            else:
                chunk_pop_idxs = None

            if using_precache:
                chunk_precached_tensors = [precached_tensors[c_idx] for
                                           c_idx in chunk_indices]
            else:
                chunk_precached_tensors = None

            processed_reqs += len(contexts)
            successful_generation = False
            retry_iter = 0
            while not successful_generation:
                try:
                    cont_toks_list = self._encode_and_generate(
                        contexts=contexts,
                        shared_tok_kwargs=shared_tok_kwargs,
                        shared_gen_kwargs=shared_gen_kwargs,
                        chunk_pop_idxs=chunk_pop_idxs,
                        chunk_precached_tensors=chunk_precached_tensors,
                        retry_iter=retry_iter,
                    )
                    successful_generation = True
                except Exception as e:
                    if is_oom_exception(e):
                        retry_iter += 1
                        print(f'WARNING: OOM exception caught {retry_iter} '
                              'time(s), retrying with different parameters '
                              'to save memory and emptying GPU cache.')
                        if self.log_misc:
                            print(e)
                            traceback.print_exc()
                        empty_gpu_cache()
                    else:
                        raise e
            for i, (cont_toks, context) in enumerate(
                    zip(cont_toks_list, contexts)):

                cont_toks = get_first_value_split(
                    seq=cont_toks, value=tok_stop_gen[0])

                s = self.tok_decode(cont_toks, skip_special_tokens=True)

                for term in stop_gen:
                    if len(term) > 0:
                        s = s.split(term)[0]

                res.append(s)
                if self.log_misc:
                    pbar.update(1)

        res = re_ords.get_original(res)

        if self.log_misc:
            pbar.close()

        return res

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing
        # than end of sentence
        return self.tokenizer.eos_token_id
