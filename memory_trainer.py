from dataclasses import dataclass, asdict
from collections import OrderedDict
from typing import Optional, Any, Dict, List, Union
import math
import json
import copy
import random
import numpy as np
import time
import os
from contextlib import nullcontext
import wandb
import torch.distributed as dist
from utils import aggregate_score_dict

from task_sampler import TaskSampler
from memory_evolution import MemoryEvolution
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from memory_llms import MemoryModelWrapper

from memory_evaluator import MemoryHFEvaluator
from memory_policy import ParamMemoryPolicy, MemoryPolicyAuxiliaryLoss
from utils import (
    COLOR, convert_to_dict_of_lists, pop_stats_from_dict_of_lists,
    concat_list_of_dicts_of_lists)


@dataclass
class WandbConfig:
    wandb_log: bool
    wandb_project: str
    wandb_run_name: str
    wandb_group_name: str


@dataclass
class TrainerConfig:
    out_dir: str
    max_iters: int

    task_batch_size: Optional[Union[int, List[int]]]

    samples_batch_size: Optional[Union[int, List[int]]]
    eval_samples_batch_size: Optional[Union[int, List[int]]]
    allow_distributed_eval: bool

    pop_accumulation_steps: int

    score_aggregation: str
    score_normalization_reference: Optional[str]

    synchronized_buffers_aggregation: str
    synchronized_buffers_freeze_after: Optional[int]

    prefetch_task_tensors: bool
    override_prefetched_tensors: bool

    eval_interval: int
    early_stop_patience: int
    log_interval: int
    eval_iters: int
    eval_only: bool

    eval_candidate_samples: Optional[int]
    eval_candidate_temp: Optional[float]

    record_advanced_eval_stats: bool
    store_eval_results_locally: bool
    record_per_task_eval_stats: bool

    always_save_checkpoint: bool
    keep_past_epoch_checkpoints_every: Optional[int]
    use_amp: Optional[bool]
    init_from: str
    dtype: str


@dataclass
class Snapshot:
    model_state: OrderedDict[str, torch.Tensor]
    optimizer_state: Dict[str, Any]
    finished_epoch: int


class MemoryTrainer():

    @torch.no_grad()
    def __init__(self,
                 device,
                 evaluation_model: MemoryHFEvaluator,
                 task_sampler: TaskSampler,
                 evolution_algorithm: MemoryEvolution,
                 trainer_config: TrainerConfig,
                 wandb_config: WandbConfig,
                 auxiliary_loss: Optional[MemoryPolicyAuxiliaryLoss] = None,
                 scratch: bool = False
                 ):

        self.evaluation_model = evaluation_model
        self.model: MemoryModelWrapper = evaluation_model.model
        self.task_sampler = task_sampler
        self.evolution_algorithm = evolution_algorithm
        self.auxiliary_loss = auxiliary_loss

        self.use_auxiliary_loss = auxiliary_loss is not None

        self.same_test_train_tasks = set(
            self.task_sampler.training_tasks_subset) == set(
                self.task_sampler.test_tasks_subset)

        self.setup_device(device=device)
        self.setup_dtype(dtype=trainer_config.dtype)

        self.model = self.model.move_model_to(dtype=self.ptdtype).to(
            device=device)
        self.evolution_algorithm = self.evolution_algorithm.to(device=device)

        self.raw_evolution_algorithm = self.evolution_algorithm

        self.out_dir = trainer_config.out_dir
        self.max_iters = trainer_config.max_iters
        self.task_batch_size = trainer_config.task_batch_size
        self.samples_batch_size = trainer_config.samples_batch_size
        self.eval_samples_batch_size = trainer_config.eval_samples_batch_size

        self.allow_distributed_eval = trainer_config.allow_distributed_eval
        self.pop_accumulation_steps = trainer_config.pop_accumulation_steps

        self.pop_size = evolution_algorithm.pop_size

        if self.world_size > self.pop_size:
            assert self.pop_accumulation_steps == 1
            assert self.world_size % self.pop_size == 0
            self.pop_batch_size = 1
            self.processes_per_pop_member = self.world_size // self.pop_size
            self.train_samples_split = True

            processes_group_idxs = np.stack(
                [np.arange(self.processes_per_pop_member) +
                 self.processes_per_pop_member*i for i in range(self.pop_size)],
                axis=0,)

            processes_group_idxs = np.repeat(
                processes_group_idxs, self.processes_per_pop_member, axis=0)

            self.all_group_processes = processes_group_idxs[self.global_rank]
            print(f'RANK {self.global_rank}: Group processes: '
                  f'{self.all_group_processes}')
        else:
            self.processes_per_pop_member = 1
            self.train_samples_split = False
            self.all_group_processes = None
            if self.pop_accumulation_steps == 1:
                assert self.pop_size % self.world_size == 0
                self.pop_batch_size = self.pop_size // self.world_size
            else:
                assert self.pop_size % self.pop_accumulation_steps == 0
                self.pop_batch_size = self.pop_size // self.pop_accumulation_steps

                assert self.pop_accumulation_steps % self.world_size == 0
                self.pop_accumulation_steps = (
                    self.pop_accumulation_steps // self.world_size)

        assert (
            self.pop_size ==
            self.pop_batch_size*self.world_size//self.processes_per_pop_member
        )

        print(f'POP batch size {self.pop_batch_size}')

        self.score_aggregation = trainer_config.score_aggregation
        assert self.score_aggregation in ['mean']

        self.score_normalization_reference = (
            trainer_config.score_normalization_reference)

        if self.score_normalization_reference is not None:
            assert os.path.isfile(self.score_normalization_reference)
            with open(self.score_normalization_reference, 'r') as f:
                self.score_normalization_reference = json.load(f)

        self.synchronized_buffers_aggregation = (
            trainer_config.synchronized_buffers_aggregation)

        assert self.synchronized_buffers_aggregation in ['mean', 'best']

        self.synchronized_buffers_freeze_after = (
            trainer_config.synchronized_buffers_freeze_after)

        if self.synchronized_buffers_freeze_after is not None:
            self.synchronized_buffers_freeze = (
                self.synchronized_buffers_freeze_after > 0)
        else:
            self.synchronized_buffers_freeze = False

        if self.synchronized_buffers_freeze == True:
            assert self.model.memory_policy_has_buffers_to_merge()

        self.eval_interval = trainer_config.eval_interval
        self.log_interval = trainer_config.log_interval

        self.early_stop_patience = trainer_config.early_stop_patience
        if self.early_stop_patience is None:
            self.early_stop_patience = -1
        self.early_stop_counter = torch.zeros(1).to(device)
        self.early_stop_flag = torch.zeros(1).to(device)

        self.eval_only = trainer_config.eval_only
        self.eval_candidate_samples = trainer_config.eval_candidate_samples
        self.eval_candidate_temp = trainer_config.eval_candidate_temp
        if self.eval_candidate_samples is not None:
            self.eval_total_samples = self.eval_candidate_samples + 1
            self.store_best_candidate_solution = True
            assert self.eval_only
            if self.eval_candidate_temp is None:
                self.eval_candidate_temp = 1.0
            if self.ddp:
                if self.eval_total_samples % self.world_size != 0:
                    if self.eval_candidate_samples % self.world_size == 0:
                        self.eval_candidate_samples -= 1
                        self.eval_total_samples -= 1
                        print('Warning - decreasing eval candidate samples to' +
                              f' {self.eval_candidate_samples} to make the ' +
                              'number of total samples divisible by the world' +
                              f' size ({self.world_size})')
                    else:
                        raise ValueError('Ensure eval_candidate_samples + 1 ' +
                                         'is divisible by the world size.')
            self.eval_params_per_ddp = self.eval_total_samples//self.world_size
            self.eval_params_idxs = torch.arange(self.eval_params_per_ddp) + (
                self.eval_params_per_ddp*self.global_rank)
            print(f'RANK {self.global_rank} - evaluation candidate idxs: ' +
                  f'{self.eval_params_idxs}')

        else:
            self.eval_total_samples = 1
            self.store_best_candidate_solution = False
            self.eval_params_per_ddp = 1

        self.record_advanced_eval_stats = (
            trainer_config.record_advanced_eval_stats)

        if self.record_advanced_eval_stats:
            self.model.memory_policy.record_eval_stats = True

        self.store_eval_results_locally = (
            trainer_config.store_eval_results_locally)
        self.record_per_task_eval_stats = (
            trainer_config.record_per_task_eval_stats)

        self.always_save_checkpoint = trainer_config.always_save_checkpoint
        self.keep_past_epoch_checkpoints_every = (
            trainer_config.keep_past_epoch_checkpoints_every)
        self.keep_all_checkpoints = False
        if self.keep_past_epoch_checkpoints_every is not None:
            self.keep_all_checkpoints = self.keep_past_epoch_checkpoints_every > 0
        if self.keep_all_checkpoints:
            assert self.always_save_checkpoint
        self.use_amp = trainer_config.use_amp
        self.init_from = trainer_config.init_from

        self.prefetch_task_tensors = trainer_config.prefetch_task_tensors
        self.override_prefetched_tensors = (
            trainer_config.override_prefetched_tensors)

        self.wandb_log = wandb_config.wandb_log
        self.wandb_project = wandb_config.wandb_project
        self.wandb_run_name = wandb_config.wandb_run_name
        self.wandb_group_name = wandb_config.wandb_group_name

        self.wandb_log = self.wandb_log and self.master_process

        params_per_ddp = self.pop_batch_size*self.pop_accumulation_steps

        params_idxs = np.arange(self.pop_batch_size) + np.expand_dims(
            np.arange(self.pop_accumulation_steps)*self.pop_batch_size, axis=1)

        self.param_idx_mx = (
            params_idxs +
            params_per_ddp*self.global_rank//self.processes_per_pop_member
        )

        print(f'RANK {self.global_rank}: pop bs {self.pop_batch_size}; ' +
              f'proc per pop {self.processes_per_pop_member}; ' +
              f'group processes {self.all_group_processes}; ' +
              f'param idx mx {self.param_idx_mx}')

        mask_vectors = []
        for i in range(self.pop_accumulation_steps):
            mask = np.zeros(self.pop_size)
            mask[self.param_idx_mx[i]] = 1
            mask_vectors.append(mask)
        self.param_idx_mx_mask = np.stack(mask_vectors, axis=0)

        if self.synchronized_buffers_aggregation == 'best':

            raise NotImplementedError

        self.start_iter = 0
        self.best_val_perf = -1e9

        self.ckpt_path = os.path.join(self.out_dir, "ckpt.pt")
        self.rng_ckpt_path = os.path.join(self.out_dir, "rng_ckpt.pt")

        self.latest_ckpt_path = os.path.join(self.out_dir, "latest.pt")
        self.numbered_ckpt_path_fmt = os.path.join(self.out_dir, "iter_{}.pt")

        self.eval_ckpt_path = os.path.join(self.out_dir, "eval_ckpt.pt")
        self.eval_path_fmt = os.path.join(self.out_dir, "eval_{}.json")

        self.scratch = scratch

        if self.prefetch_task_tensors:
            self.task_sampler.prefetch_model_tensors(
                lm=self.evaluation_model,
                lm_name=self.evaluation_model.model_name,
                limit=None,
                override=False,
            )

        self.force_initial_re_eval = False

        if os.path.isfile(self.latest_ckpt_path) and not scratch:
            assert self.always_save_checkpoint

            success = self._load_ckpt(
                load_randomness=False, load_path=self.latest_ckpt_path)
            if success:
                print(f'PROCESS {self.global_rank}: successfully resuming ' +
                      f'from checkpoint {self.latest_ckpt_path}')
            else:
                print('WARNING: unable to initialize model from specified ' +
                      f'path {self.latest_ckpt_path}, restarting from scratch.')
        elif self.init_from is not None:
            success = False
            if os.path.isfile(self.init_from):
                success = self._load_ckpt(
                    load_randomness=False, load_path=self.init_from)
            if not success:
                print('WARNING: unable to initialize model from specified ' +
                      f'path {self.init_from}, restarting from scratch.')
                raise ValueError()
            else:
                print(f'PROCESS {self.global_rank}: successfully resuming ' +
                      f'from checkpoint {self.init_from}')
                self.evolution_algorithm.shift_first_gen = False
            self.force_initial_re_eval = True

    @torch.no_grad()
    def aggregate_score_dict(self, score_dict, sample_idxs_per_task=None):
        score, norm_score_dict = aggregate_score_dict(
            score_dict=score_dict,
            score_aggregation=self.score_aggregation,
            score_normalization_reference=self.score_normalization_reference,
            sample_idxs_per_task=sample_idxs_per_task,
            task_names=None,
        )
        return score, norm_score_dict

    @torch.no_grad()
    def aggregate_scores(self, score_dicts, sample_idxs_per_task=None):
        aggregated_scores = []
        norm_score_dicts = []
        for score_dict in score_dicts:
            score, norm_score_dict = self.aggregate_score_dict(
                score_dict=score_dict,
                sample_idxs_per_task=sample_idxs_per_task)
            aggregated_scores.append(score)
            norm_score_dicts.append(norm_score_dict)
        return aggregated_scores, norm_score_dicts

    def sample_and_synchronize_task_idxs(
            self, train=True, split=False, sampled_requests=None,
            task_batch_size=None, reshuffle=False, num_splits=None):

        if self.master_process:

            self.task_sampler.resample_requests(
                train=train,
                sampled_requests_per_task=sampled_requests,
                task_batch_size=task_batch_size,
            )
            latest_sampled_task_idxs = (
                self.task_sampler.get_latest_sampled_idxs(train=train))

        else:
            latest_sampled_task_idxs = None

        if self.ddp:

            if num_splits is None:
                num_splits = self.world_size

            task_idxs_to_scatter = [None for _ in range(self.world_size)]
            if self.master_process:
                if split:
                    assert self.world_size % num_splits == 0
                    num_groups = self.world_size//num_splits

                    split_allocation = np.concatenate(
                        [np.arange(num_splits) for _ in range(num_groups)],
                        axis=0,
                    )
                    task_idxs_to_scatter = [{} for _ in range(self.world_size)]
                    for task_n, task_idxs in latest_sampled_task_idxs.items():
                        if reshuffle:
                            task_idxs_to_split = np.random.permutation(
                                task_idxs)
                        else:
                            task_idxs_to_split = task_idxs
                        split_idxs = np.array_split(
                            task_idxs_to_split, num_splits)

                        if reshuffle:
                            num_split_idxs = len(split_idxs)
                            split_idxs = [
                                split_idxs[i] for i in
                                np.random.permutation(num_split_idxs)
                            ]

                        for process_idx in range(self.world_size):
                            split_partition_idx = split_allocation[process_idx]
                            task_idxs_to_scatter[process_idx][task_n] = (
                                split_idxs[split_partition_idx])
                else:
                    task_idxs_to_scatter = [latest_sampled_task_idxs for _
                                            in range(self.world_size)]

            receiving_output_idxs = [None]

            dist.scatter_object_list(receiving_output_idxs,
                                     task_idxs_to_scatter, src=0)

            latest_sampled_task_idxs = receiving_output_idxs[0]

            self.task_sampler.set_requests_per_task(latest_sampled_task_idxs)

    @torch.no_grad()
    def get_best_candidate_and_synchronize_evolution(
        self,
        fitness: torch.Tensor,
        candidate_samples: torch.Tensor,
        process_results_dicts: Optional[dict] = None,
    ):

        if self.ddp:
            dist.all_reduce(fitness, op=dist.ReduceOp.SUM)
        best_fitness, best_candidate_idx = torch.max(fitness, dim=0)
        best_results_dict = None

        best_candidate_idx = best_candidate_idx.item()
        best_candidate_process = best_candidate_idx // self.eval_params_per_ddp
        best_candidate_process_idx = (
            best_candidate_idx % self.eval_params_per_ddp)

        if self.master_process:
            best_candidate = candidate_samples[best_candidate_idx]
            buffers = self.model.get_buffers_dict()

            self.evolution_algorithm.store_best_params(
                x=best_candidate, fitness=best_fitness)
            self.evolution_algorithm.store_best_buffers(buffers=buffers)

        if self.ddp:
            self.synchronize_evolution_from_master()

        if process_results_dicts is not None:
            if self.ddp:
                results_dicts_per_process = self.gather_objects_to(
                    obj=process_results_dicts, dst=0)
                if self.master_process:
                    best_results_dict = results_dicts_per_process[
                        best_candidate_process][best_candidate_process_idx]
            else:
                assert best_candidate_process == 0
                best_results_dict = process_results_dicts[
                    best_candidate_process_idx]
                results_dicts_per_process = [process_results_dicts]

        if self.master_process:
            for el in (best_candidate_idx, best_candidate_process,
                       best_candidate_process_idx, best_results_dict):
                print(f'Master: {el}')
        return (best_candidate_idx, best_candidate_process,
                best_candidate_process_idx, best_results_dict,
                results_dicts_per_process, best_fitness, fitness)

    def log_score_dict(self, prefix, sample_idxs_per_task, score_dict,
                       suffix='',):
        out = {}
        mean_score, norm_score_dict = self.aggregate_score_dict(
            score_dict=score_dict,
            sample_idxs_per_task=sample_idxs_per_task,
        )
        out[prefix + '_tasks_aggregate'] = mean_score
        for task_name, task_score in score_dict.items():
            if isinstance(task_score, dict):
                continue
            out[prefix + '_' + task_name] = task_score
            if self.score_normalization_reference is not None:
                out[prefix + '_' + task_name + '_norm'] = (
                    norm_score_dict[task_name])
        return out

    @torch.no_grad()
    def update_and_synchronize_evolution(
        self,
        fitness: torch.Tensor,
    ):

        if self.model.memory_policy_has_buffers_to_merge():
            self.merge_and_store_buffers()

        if self.ddp:

            dist.all_reduce(fitness, op=dist.ReduceOp.SUM)

        if self.master_process:

            self.evolution_algorithm.tell(
                fitness=fitness/self.processes_per_pop_member)

        if self.ddp:
            self.synchronize_evolution_from_master()

    @torch.no_grad()
    def synchronize_evolution_from_master(self,):
        if self.ddp:

            for n, p in self.raw_evolution_algorithm.named_parameters():

                if self.master_process:

                    scatter_list = [p for _ in range(self.world_size)]
                else:
                    scatter_list = None
                dist.scatter(p.data, scatter_list, src=0)

    @torch.no_grad()
    def merge_and_store_buffers(
        self,
        target_buffer_list: Optional[List[torch.tensor]] = None,
    ):

        if self.model.memory_policy_has_buffers_to_merge() and (
                not self.model.are_sync_buffers_frozen()):
            if self.ddp:
                tensor_list_to_merge = self.model.get_buffers_list()

                tensor_list_of_lists = []
                for i, tensor in enumerate(tensor_list_to_merge):
                    if self.master_process:
                        gathered_tensor_list = [torch.zeros_like(
                            tensor) for _ in range(self.world_size)]
                        dist.gather(
                            tensor=tensor,
                            gather_list=gathered_tensor_list,
                            dst=0,
                        )
                        if target_buffer_list is not None:
                            gathered_tensor_list = [
                                target_buffer_list[i]] + gathered_tensor_list
                        tensor_list_of_lists.append(gathered_tensor_list)
                    else:
                        dist.gather(tensor=tensor, dst=0,)

                if self.master_process:
                    merged_tensor_list = self.model.merge_buffers_list(
                        buffers_to_merge=tensor_list_of_lists,
                    )
                else:
                    merged_tensor_list = [torch.zeros_like(
                        tensor) for tensor in tensor_list_to_merge]

                for tensor in merged_tensor_list:
                    dist.broadcast(tensor=tensor, src=0)

            else:
                if target_buffer_list is None:
                    merged_tensor_list = self.model.self_merge()
                else:
                    raise NotImplementedError

            self.model.receive_buffers_list(buffers_list=merged_tensor_list)

            buffers_to_save = self.model.get_buffers_dict()
            self.raw_evolution_algorithm.store_buffers(
                buffers=buffers_to_save)

    @torch.no_grad()
    def sample_and_synchronize_params(self, best=False):
        if best:
            best_params = self.evolution_algorithm.best_params.unsqueeze(0)
            if self.eval_only and (self.eval_candidate_samples is not None):
                candidate_params = self.evolution_algorithm.sample_candidates(
                    num_candidates=self.eval_candidate_samples,
                    temperature=self.eval_candidate_temp,
                )
                step_params = torch.concat(
                    [best_params, candidate_params], dim=0)

                if self.eval_only and self.model.memory_policy.lazy_param_num:
                    print('Setting stored lazy parameter number to ' +
                          f'{self.eval_total_samples} for candidate evaluation')
                elif self.eval_total_samples < self.pop_size:
                    padding_params = self.pop_size - self.eval_total_samples
                    filler_params = torch.zeros_like(best_params).expand(
                        padding_params, -1)
                    step_params = torch.concat(
                        [step_params, filler_params], dim=0)
                else:
                    assert (self.eval_total_samples == self.pop_size
                            ), ('Candidate eval. with more total samples than' +
                                ' pop_size not implemented')
            else:
                step_params = best_params.expand(self.pop_size, -1)
        else:
            step_params = self.evolution_algorithm.ask()

        step_params = step_params.contiguous()
        if self.ddp:

            dist.broadcast(tensor=step_params, src=0)

        if self.model.memory_policy_has_buffers_to_merge():

            buffers = self.evolution_algorithm.get_stored_buffers()
        else:
            buffers = {}
        return step_params, buffers

    def setup_device(self, device):
        self.device = device
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'

        self.global_rank = int(os.environ.get('RANK', -1))
        self.ddp = self.global_rank > -1

        if self.ddp:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.local_rank}'
            torch.cuda.set_device(self.device)

            self.master_process = self.global_rank == 0

            print(f'Initializing trainer for process {self.global_rank} '
                  f'(is master={self.master_process} '
                  f'world_size={self.world_size})')

            assert self.device_type == 'cuda'

        else:
            self.local_rank = 0
            self.global_rank = 0
            self.master_process = True
            self.seed_offset = 0
            self.world_size = 1

    def setup_dtype(self, dtype):
        self.dtype = dtype
        self.ptdtype = {'float32': torch.float32,
                        'bfloat16': torch.bfloat16,
                        'float16': torch.float16}[self.dtype]

        print(f'PTDType used {self.ptdtype}')

    @torch.no_grad()
    def gather_params(self, pop_accumulation_idx: int, ask_new: bool,):
        if ask_new:
            self.pop_params = self.evolution_algorithm.ask()
        idxs = self.param_idx_mx[pop_accumulation_idx].unsqueeze(-1).expand(
            -1, self.pop_params.shape[-1])

        return torch.gather(self.pop_params, dim=0, index=idxs)

    @torch.no_grad()
    def estimate_loss(self,):
        '''Estimate loss on sequentially sampled data, if iid go back
           defaults to non-sequential.'''
        out = {}
        candidate_out = {}
        self.model.eval()
        self.model.evaluation_mode()
        if self.use_auxiliary_loss:
            self.auxiliary_loss.restart_recording()

        cache_eval_stats_per_task = (self.record_per_task_eval_stats and
                                     self.master_process)
        is_distributed = self.allow_distributed_eval and self.ddp
        params, buffers = self.sample_and_synchronize_params(best=True)

        self.model.set_memory_params(params=params)

        if self.model.memory_policy_has_buffers_to_merge():
            self.model.load_buffers_dict(buffers_dict=buffers)

        if self.same_test_train_tasks:
            splits = ['train']
        else:
            splits = ['train', 'val']
        for split in splits:
            train = split == 'train'

            if not train:
                if self.model.memory_policy_has_buffers_to_merge():
                    frozen_state = self.model.are_sync_buffers_frozen()
                    self.model.freeze_sync_buffers(freeze=True)
            if self.eval_candidate_samples is not None and train:
                idx_sampling_kwargs = dict(split=False)
                evaluation_kwargs = dict(
                    pop_reps=self.eval_params_per_ddp,
                    pop_idxs=self.eval_params_idxs)
                candidate_scores = torch.zeros(
                    [self.eval_total_samples], device=self.device)
            else:
                idx_sampling_kwargs = dict(split=True)
                evaluation_kwargs = dict(
                    pop_reps=1,
                    pop_idxs=None,
                )
            if is_distributed:

                self.sample_and_synchronize_task_idxs(
                    train=train,
                    sampled_requests=self.eval_samples_batch_size,
                    reshuffle=True,
                    **idx_sampling_kwargs,
                )
                score_dicts = self.task_sampler.evaluate(
                    lm=self.evaluation_model,
                    train=True,
                    evolved_model=True,
                    resample_requests=False,
                    performance_per_request=True,
                    cache_param_stats_per_task=cache_eval_stats_per_task,
                    **evaluation_kwargs,


                )
            else:
                score_dicts = self.task_sampler.evaluate(
                    lm=self.evaluation_model,
                    train=train,
                    evolved_model=True,
                    cache_param_stats_per_task=cache_eval_stats_per_task,



                    sampled_requests_per_task=self.eval_samples_batch_size,
                    **evaluation_kwargs,
                )
            sample_idxs_per_task = self.task_sampler.get_latest_sampled_idxs()
            if self.eval_candidate_samples is not None and train:
                scores, norm_score_dicts = self.aggregate_scores(
                    score_dicts=score_dicts,
                    sample_idxs_per_task=sample_idxs_per_task,
                )
                candidate_scores[self.eval_params_idxs] = torch.tensor(
                    scores, device=self.device, dtype=candidate_scores.dtype)
                if self.use_auxiliary_loss:
                    raise NotImplementedError
                else:
                    fitness = candidate_scores
                (best_candidate_idx, best_candidate_process,
                 best_candidate_process_idx, best_results_dict,
                 results_dicts_per_process, best_fitness, fitness) = (
                     self.get_best_candidate_and_synchronize_evolution(
                         fitness=fitness,
                         candidate_samples=params,
                         process_results_dicts=score_dicts,
                     )
                )
                if self.master_process:
                    score_dict = best_results_dict
                    candidate_idx = 0

                    for process_dicts in results_dicts_per_process:
                        for process_score_dict in process_dicts:
                            candidate_out.update(self.log_score_dict(
                                prefix=f'candidate_{candidate_idx}/train',
                                sample_idxs_per_task=sample_idxs_per_task,
                                score_dict=process_score_dict,
                                suffix='',
                            ))
                            candidate_idx += 1
                    candidate_out['candidate_stats/best_idx'] = (
                        best_candidate_idx)
                    candidate_out['candidate_stats/best_fitness'] = (
                        best_fitness.item())
                    candidate_out['candidate_stats/std_fitness'] = (
                        torch.std(fitness).item())
                    candidate_out['candidate_stats/min_fitness'] = (
                        torch.min(fitness).item())
                    candidate_out['candidate_stats/mean_fitness'] = (
                        torch.mean(fitness).item())
                else:
                    score_dict = score_dicts[0]
            else:
                score_dict = score_dicts[0]
                if not train:
                    if self.model.memory_policy_has_buffers_to_merge():
                        self.model.freeze_sync_buffers(freeze=frozen_state)
                if is_distributed:
                    score_dict_list = self.gather_objects_to(
                        obj=score_dict, dst=0)
                    if self.master_process:
                        (prompts_per_task, scores_per_task,
                         mean_scores_per_task) = self.merge_task_results(
                             list_of_stats=score_dict_list)
                        score_dict = mean_scores_per_task
                        sample_idxs_per_task = prompts_per_task

            out.update(self.log_score_dict(
                prefix=split,
                sample_idxs_per_task=sample_idxs_per_task,
                score_dict=score_dict,
                suffix='',
            ))

            if self.use_auxiliary_loss:
                aux_loss = self.auxiliary_loss.get_loss()[0]
                if is_distributed:
                    aux_loss = self.reduce_tensors_to(
                        tensor=aux_loss, dst=0, mean=True)
                if self.master_process:
                    out[split + '_aux_loss'] = aux_loss.item()

        if self.same_test_train_tasks:
            split = 'val'
            val_out = {}
            for k, v in out.items():
                task_name = k.removeprefix('train_')
                val_out[split + '_' + task_name] = v
            out.update(val_out)

        if self.master_process and (self.eval_candidate_samples is not None):
            out.update(candidate_out)

        return out

    @torch.no_grad()
    def synchronize_dict(self, input_dict):
        if self.ddp:

            synced_dict = {}
            for key, value in input_dict.items():
                if isinstance(value, dict):
                    continue
                data_tensor = torch.tensor(value, device=self.device)
                dist.all_reduce(data_tensor, op=dist.ReduceOp.SUM)
                synced_dict[key] = (data_tensor/self.world_size).item()
        else:
            synced_dict = input_dict
        return synced_dict

    @torch.no_grad()
    def gather_objects_to(self, obj, dst=0):
        if self.ddp:
            if dist.get_rank() == dst:
                synced_list = [None for _ in range(self.world_size)]
            else:
                synced_list = None
            dist.gather_object(
                obj=obj,
                object_gather_list=synced_list,
                dst=dst,
            )
        else:
            synced_list = [obj]
        return synced_list

    @torch.no_grad()
    def gather_objects_all(self, obj):
        if self.ddp:
            synced_list = [None for _ in range(self.world_size)]
            dist.all_gather_object(object_list=synced_list, obj=obj)
        else:
            synced_list = [obj]
        return synced_list

    @torch.no_grad()
    def merge_task_results(self, list_of_stats):
        prompts_per_task = {}
        scores_per_task = {}
        mean_scores_per_task = {}

        for process_stats in list_of_stats:
            for task_n, task_score_dict in process_stats[
                    'performance_per_request'].items():
                current_prompts = prompts_per_task.get(task_n, [])
                current_scores = scores_per_task.get(task_n, [])

                current_prompts += list(task_score_dict.keys())
                current_scores += list(task_score_dict.values())
                prompts_per_task[task_n] = current_prompts
                scores_per_task[task_n] = current_scores

        for task_n, scores in scores_per_task.items():
            scores_per_task[task_n] = np.array(scores)
            mean_scores_per_task[task_n] = np.mean(scores_per_task[task_n])*100
            prompts_per_task[task_n] = np.array(
                prompts_per_task[task_n]).astype(int)

            if not (np.unique(prompts_per_task[task_n]).shape[0] ==
                    prompts_per_task[task_n].shape[0]):
                raise ValueError('ERROR: Repeated prompt indexes found when ' +
                                 f'merging results for task {task_n}')
        return prompts_per_task, scores_per_task, mean_scores_per_task

    @torch.no_grad()
    def reduce_tensors_to(self, tensor, dst=0, mean=True):
        if self.ddp:
            dist.reduce(tensor, dst=dst, op=dist.ReduceOp.SUM)
            if mean:
                tensor = tensor/self.world_size
        return tensor

    @torch.no_grad()
    def _evaluate(self, iter_num, **log_kwargs):

        if self.use_auxiliary_loss:
            self.auxiliary_loss.restart_recording()

        if self.master_process or self.allow_distributed_eval:
            evaluation_results = self.estimate_loss()

        if self.model.memory_policy_has_buffers_to_merge() and (
                not self.model.are_sync_buffers_frozen()):
            self.merge_and_store_buffers()

        wandb_log_dict = {}
        if self.master_process:
            if self.record_per_task_eval_stats:
                memory_policy_stats = (
                    self.task_sampler.get_cached_per_task_stats())
            else:
                memory_policy_stats = self.model.get_param_stats()
            evo_stats = self.evolution_algorithm.get_stats()
            log_str = COLOR.LIGHT_CYAN + f"step {iter_num}"
            for k, v in evaluation_results.items():
                log_str += f" | {k}:{v:.4f}"
            print(log_str)
            wandb_log_dict = {
                "iter": iter_num,
                **evaluation_results,
                **log_kwargs,
                **memory_policy_stats,
                **evo_stats,
            }
            val_perf = evaluation_results['val_tasks_aggregate']

            if self.wandb_log:
                wandb.log(wandb_log_dict)
            if self.store_best_candidate_solution and self.master_process:
                self._save_ckpt(
                    iter_num=iter_num, save_path=self.eval_ckpt_path)
            if val_perf > self.best_val_perf:
                self.best_val_perf = val_perf
                if iter_num > self.start_iter:
                    if self.master_process:
                        self._save_ckpt(
                            iter_num=iter_num, save_path=self.ckpt_path)
                self.early_stop_counter = self.early_stop_counter.zero_()
            elif self.early_stop_patience > 0:
                self.early_stop_counter = self.early_stop_counter.add_(1)
                self.early_stop_flag = (self.early_stop_counter >
                                        self.early_stop_patience)
        return wandb_log_dict

    @torch.no_grad()
    def _train_step(self,):
        self.model.training_mode()

        step_params, buffers = self.sample_and_synchronize_params(best=False)
        self.model.set_memory_params(step_params)

        if self.model.memory_policy_has_buffers_to_merge():
            self.model.load_buffers_dict(buffers_dict=buffers)

        step_scores = torch.zeros([self.pop_size], device=self.device)

        all_score_dicts = []
        all_scores = []
        self.sample_and_synchronize_task_idxs(
            train=True,
            split=self.train_samples_split,
            sampled_requests=self.samples_batch_size,
            task_batch_size=self.task_batch_size,
            num_splits=self.processes_per_pop_member,
        )
        if self.use_auxiliary_loss:
            self.auxiliary_loss.restart_recording()
        score_dicts_per_acc_step = []
        for pop_acc_step in range(self.pop_accumulation_steps):
            acc_step_idxs = self.param_idx_mx[pop_acc_step]

            score_dicts = self.task_sampler.evaluate(
                lm=self.evaluation_model, train=True, evolved_model=True,
                resample_requests=False,
                pop_reps=self.pop_batch_size, pop_idxs=acc_step_idxs,
                sampled_requests_per_task=self.samples_batch_size,


                performance_per_request=self.train_samples_split,
            )
            score_dicts_per_acc_step.append(score_dicts)

        if self.train_samples_split:

            all_score_dicts = self.gather_objects_all(obj=score_dicts[0])
            all_group_score_dicts = [
                all_score_dicts[score_dicts_idx] for score_dicts_idx
                in self.all_group_processes]

            (prompts_per_task, scores_per_task,
             mean_scores_per_task) = self.merge_task_results(
                list_of_stats=all_group_score_dicts)
            score_dict = mean_scores_per_task
            sample_idxs_per_task = prompts_per_task
            score_dicts_per_acc_step = [[score_dict]]

        else:
            sample_idxs_per_task = self.task_sampler.get_latest_sampled_idxs()

        for pop_acc_step, score_dicts in enumerate(score_dicts_per_acc_step):
            acc_step_idxs = self.param_idx_mx[pop_acc_step]

            scores, norm_score_dicts = self.aggregate_scores(
                score_dicts=score_dicts,
                sample_idxs_per_task=sample_idxs_per_task,
            )

            if self.score_normalization_reference is not None:
                for idx_p, (score_dict_p, norm_score_dict_p) in enumerate(
                        zip(score_dicts, norm_score_dicts)):
                    for task_n, task_score_n in norm_score_dict_p.items():
                        if isinstance(task_score_n, dict):
                            continue
                        score_dicts[idx_p][task_n + '_norm'] = task_score_n
            all_score_dicts += score_dicts
            all_scores += scores
            step_scores[acc_step_idxs] = torch.tensor(
                scores, device=self.device, dtype=step_scores.dtype)

        if self.use_auxiliary_loss:
            aux_loss = self.auxiliary_loss()
            fitness = step_scores - aux_loss
        else:
            fitness = step_scores
        self.update_and_synchronize_evolution(fitness=fitness)
        return all_scores, all_score_dicts

    def _save_ckpt(self, iter_num, save_path=None):
        if save_path == None:
            ckpt_path = self.ckpt_path
            rng_ckpt_path = self.rng_ckpt_path
        else:
            ckpt_path = save_path
            basename = 'rng_' + os.path.basename(ckpt_path)
            dirname = os.path.dirname(ckpt_path)
            rng_ckpt_path = os.path.join(dirname, basename)
        checkpoint = {
            'evolution_state': self.raw_evolution_algorithm.state_dict(),
            'iter_num': iter_num,
            'best_val_loss': self.best_val_perf,
        }
        rng_checkpoint = {
            'cpu_rng_state': torch.get_rng_state(),
            'gpu_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'py_rng_state': random.getstate()
        }
        print(f"saving checkpoint to {ckpt_path}")
        ckpt_dir = os.path.dirname(ckpt_path)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        torch.save(checkpoint, ckpt_path)
        print(f"saving checkpoint to {rng_ckpt_path}")
        torch.save(rng_checkpoint, rng_ckpt_path)

    def _load_ckpt(self, load_randomness, load_path=None):
        if load_path == None:
            ckpt_path = self.ckpt_path
            rng_ckpt_path = self.rng_ckpt_path
        else:
            ckpt_path = load_path
            basename = 'rng_' + os.path.basename(ckpt_path)
            dirname = os.path.dirname(ckpt_path)
            rng_ckpt_path = os.path.join(dirname, basename)

        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state_dict = checkpoint['evolution_state']

            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            print(f"Loading checkpoint from {ckpt_path}")
            if self.model.memory_policy_has_buffers_to_merge():
                buffers_prefix = 'stored_buffers_to_save.'
                buffers_dict = {}
                for k, v in list(state_dict.items()):
                    if k.startswith(buffers_prefix):
                        buffers_dict[k[len(buffers_prefix):]] = v
                self.raw_evolution_algorithm.store_buffers(
                    buffers=buffers_dict, best=False)

                state_dict_keys = list(state_dict.keys())
                best_buffers_prefix = 'best_stored_buffers_to_save.'
                has_best_buffers = any(
                    [k.startswith(best_buffers_prefix)]
                    for k in state_dict_keys)
                if not has_best_buffers:
                    buffers_dict = {}
                    for k, v in list(state_dict.items()):
                        if k.startswith(best_buffers_prefix):
                            buffers_dict[k[len(best_buffers_prefix):]] = v

                else:

                    print('WARNING, legacy ckpt - loading best stored buffers' +
                          ' using non-best stored buffers keys')
                    for k, v in buffers_dict.items():
                        state_dict[best_buffers_prefix+k] = v
                self.raw_evolution_algorithm.store_buffers(
                    buffers=buffers_dict, best=True)
                print('Initialized buffers')

            print('Pre loading: best population member')
            print(self.raw_evolution_algorithm.best_member)
            self.raw_evolution_algorithm.load_state_dict(state_dict)
            print('Post loading: best population member')
            print(self.raw_evolution_algorithm.best_member)
            if load_randomness:
                rng_state_dict = torch.load(rng_ckpt_path,
                                            map_location='cpu')
                torch.set_rng_state(rng_state_dict['cpu_rng_state'])
                torch.cuda.set_rng_state(rng_state_dict['gpu_rng_state'])
                np.random.set_state(rng_state_dict['numpy_rng_state'])
                random.setstate(rng_state_dict['py_rng_state'])
            self.start_iter = checkpoint['iter_num'] + 1
            return True
        return False

    @torch.no_grad()
    def train(self):
        local_iter_num = 0
        t0 = time.time()
        if self.eval_only:
            start_iter = 0
            max_iters = 0
        else:
            start_iter = self.start_iter
            max_iters = self.max_iters

        for iter_num in range(start_iter, max_iters+1):

            if self.model.memory_policy_has_buffers_to_merge:
                if self.synchronized_buffers_freeze:
                    if iter_num >= self.synchronized_buffers_freeze_after:
                        frozen_state = self.model.are_sync_buffers_frozen()
                        if not frozen_state:
                            print('Freezing memory policy synchronized buffers')
                            self.model.freeze_sync_buffers(freeze=True)
                    elif self.eval_only:
                        print('Freezing memory policy synchronized buffers ' +
                              'for evaluation-only run')
                        self.model.freeze_sync_buffers()

            if ((iter_num % self.eval_interval == 0) or
                    self.eval_only) or (self.force_initial_re_eval and
                                        iter_num == self.start_iter):

                logged_dict = self._evaluate(iter_num=iter_num,)
                if self.store_eval_results_locally:
                    if self.master_process:
                        store_path = self.eval_path_fmt.format(iter_num)
                        directory = os.path.dirname(store_path)
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        with open(store_path, 'w') as f:
                            json.dump(logged_dict, f, indent=2)
                if self.early_stop_patience > 0:
                    if self.ddp:
                        early_stop_flag = dist.all_reduce(self.early_stop_flag)
                    else:
                        early_stop_flag = self.early_stop_flag
                    if early_stop_flag.item() > 0:
                        print(f'Terminating due to no improvement in '
                              f'{self.early_stop_patience} steps.')
                        break
            if self.eval_only:
                return logged_dict
            all_scores, all_scores_dict = self._train_step()
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % self.log_interval == 0:
                aggregated_result_dict = convert_to_dict_of_lists(
                    all_scores_dict)
                aggregated_scores_dict = dict(tasks_aggregate=all_scores)
                if self.ddp:
                    aggregated_result_dict_list = self.gather_objects_to(
                        obj=aggregated_result_dict, dst=0)
                    aggregate_scores_dict_list = self.gather_objects_to(
                        obj=aggregated_scores_dict, dst=0)
                    if self.master_process:
                        aggregated_result_dict = concat_list_of_dicts_of_lists(
                            result_dicts_list=aggregated_result_dict_list)
                        aggregated_scores_dict = concat_list_of_dicts_of_lists(
                            result_dicts_list=aggregate_scores_dict_list)
                if self.master_process:
                    log_dict = pop_stats_from_dict_of_lists(
                        aggregated_result_dict=aggregated_result_dict,
                        prefix='pop/')
                    log_dict_scores = pop_stats_from_dict_of_lists(
                        aggregated_result_dict=aggregated_scores_dict,
                        prefix='pop/')
                    log_str = (COLOR.GREEN + f"POP STATS - step {iter_num}" +
                               f" | time {dt*1000:.2f}ms")
                    for k, v in log_dict_scores.items():
                        log_str += f" | {k}:{v:.4f}"
                    print(log_str)
                    if self.wandb_log:
                        memory_policy_stats = self.model.get_param_stats()
                        evo_stats = self.evolution_algorithm.get_stats()
                        wandb_log_dict = {
                            "iter": iter_num,
                            **log_dict,
                            **log_dict_scores,
                            **memory_policy_stats,
                            **evo_stats,
                        }
                        wandb.log(wandb_log_dict)

            if self.master_process:
                if self.always_save_checkpoint and iter_num > self.start_iter:
                    self._save_ckpt(iter_num=iter_num,
                                    save_path=self.latest_ckpt_path)
                    if self.keep_all_checkpoints:
                        if (iter_num % self.keep_past_epoch_checkpoints_every
                                == 0):
                            self._save_ckpt(
                                iter_num=iter_num,
                                save_path=self.numbered_ckpt_path_fmt.format(
                                    iter_num),
                            )
            local_iter_num += 1
        iter_num += 1
