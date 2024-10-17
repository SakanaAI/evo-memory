import os
import random
import itertools
import json
import collections
import logging
import sys
from typing import Optional, Union
from datasets import load_dataset

import json
import copy

import numpy as np

from utils_longbench import get_score


def merge_list_of_dicts(dicts, other_dicts):
    assert len(dicts) == len(other_dicts)
    new_dicts = [{**d1, **d2} for d1, d2 in zip(dicts, other_dicts)]
    return new_dicts


class TaskSampler():
    def __init__(
        self,
        tasks,  # list of tasks to load
        metrics,  # list of metrics per task
        training_tasks_subset: Optional[list] = None,
        test_tasks_subset: Optional[list] = None,
        store_gen_outputs: bool = False,
        store_gen_outputs_path: Optional[str] = None,
    ):

        self.store_gen_outputs = store_gen_outputs

        if store_gen_outputs_path is not None:
            assert store_gen_outputs
        elif store_gen_outputs:
            store_gen_outputs_path = 'generated_outputs/temp/'

        self.store_gen_outputs_path = store_gen_outputs_path

        if store_gen_outputs:
            if not os.path.exists(store_gen_outputs_path):
                os.makedirs(store_gen_outputs_path)

        if type(tasks) == str:
            tasks = [tasks]
        else:
            tasks = list(tasks)
        if type(metrics) == str:
            metrics = [metrics for _ in tasks]
        else:
            metrics = list(metrics)

        assert len(metrics) == len(tasks)

        self.lb_tasks = []
        self.lb_metrics = []
        self.lb_datasets = []
        for t, m in zip(tasks, metrics):
            if t.startswith('lb/'):
                self.add_long_bench_task(task=t, metric=m)
            elif t.startswith('choubun/'):
                self.add_choubun_task(task=t, metric=m)
            else:
                raise NotImplementedError

        self.training_tasks_subset = training_tasks_subset or tasks
        self.test_tasks_subset = test_tasks_subset or tasks
        self.prefetched_task_tensors = {t: None for t in tasks}
        self.loaded_cached_model_data = False
        self.cached_per_task_stats = {}
        self.init_tasks()

    def get_cached_per_task_stats(self, reset=True) -> dict:
        cached_per_task_stats = copy.deepcopy(self.cached_per_task_stats)
        if reset:
            self.cached_per_task_stats = {}
        return cached_per_task_stats

    def add_long_bench_task(self, task, metric):
        bench_name, task_name = task.split('/')
        assert bench_name == 'lb'
        dataset = load_dataset('THUDM/LongBench', task_name, split='test')
        self.lb_datasets.append(dataset)
        self.lb_tasks.append(task)
        self.lb_metrics.append(metric)

    def add_choubun_task(self, task, metric):
        bench_name, task_name = task.split('/')
        assert bench_name == 'choubun'
        dataset = load_dataset('SakanaAI/ChouBun', task_name, split='test')
        self.lb_datasets.append(dataset)
        self.lb_tasks.append(task)
        self.lb_metrics.append(metric)

    def init_tasks(self,):
        # LongBench
        longbench_path = 'LongBench/'
        self.lb_task2prompt = json.load(open(
            f"{longbench_path}config/dataset2prompt.json", "r"))
        self.lb_task2prompt = {'lb/' + t: v
                               for t, v in self.lb_task2prompt.items()}
        self.lb_task2maxlen = json.load(open(
            f"{longbench_path}config/dataset2maxlen.json", "r"))
        self.lb_task2maxlen = {'lb/' + t: v
                               for t, v in self.lb_task2maxlen.items()}
        self.lb_taskstopgen = {t: [] for t in self.lb_task2maxlen}
        self.lb_taskstopgen["lb/samsum"].append('\n')

        # ChouBun
        choubun_path = 'ChouBun/'
        choubun_task2prompt = json.load(open(
            f"{choubun_path}config/dataset2prompt.json", "r"))
        choubun_task2prompt = {'choubun/' + t: v
                                 for t, v in choubun_task2prompt.items()}
        choubun_task2maxlen = json.load(open(
            f"{choubun_path}config/dataset2maxlen.json", "r"))
        choubun_task2maxlen = {'choubun/' + t: v
                               for t, v in choubun_task2maxlen.items()}
        choubun_taskstopgen = {t: [] for t in choubun_task2maxlen}
        self.lb_task2prompt.update(choubun_task2prompt)
        self.lb_task2maxlen.update(choubun_task2maxlen)
        self.lb_taskstopgen.update(choubun_taskstopgen)

        self.lb_dataset_per_task = {t: d for t, d in zip(
            self.lb_tasks, self.lb_datasets)}

        # unpacked utils
        self.lb_jsons_per_task = {t: [p for p in d] for t, d in zip(
            self.lb_tasks, self.lb_datasets)}

        self.lb_prompts_per_task = {}
        for task, jsons in self.lb_jsons_per_task.items():
            prompt_format = self.lb_task2prompt[task]
            self.lb_prompts_per_task[task] = []
            for json_file in jsons:
                prompt = prompt_format.format(**json_file)
                self.lb_prompts_per_task[task].append(prompt)

        self.num_prompts_per_lb_task = {k: len(
            ps) for k, ps in self.lb_prompts_per_task.items()}

        self.latest_sampled_idxs_per_lb_task = None
        self.latest_lb_tasks_names = None

        self.lb_training_tasks = [t for t in self.lb_tasks
                                  if t in self.training_tasks_subset]
        self.lb_test_tasks = [t for t in self.lb_tasks
                              if t in self.test_tasks_subset]

    def resample_requests(self, train: bool,
                          sampled_requests_per_task: Optional[int] = None,
                          task_batch_size: Optional[int] = None,
                          ) -> None:
        self.resample_requests_lb(
            train=train,
            sampled_requests_per_task=sampled_requests_per_task,
            task_batch_size=task_batch_size)

    def set_requests_per_task(self, requests_dict):
        self.latest_lb_tasks_names = []
        self.latest_sampled_idxs_per_lb_task = {}

        for task_n, task_idxs in requests_dict.items():
            if task_n in self.lb_tasks:
                self.latest_lb_tasks_names.append(task_n)
                self.latest_sampled_idxs_per_lb_task.update(
                    {task_n: task_idxs})
            else:
                raise ValueError(
                    'Invalid task name passed when setting task idxs')

    def get_requests_per_task(self,):
        out_dict = {}
        out_dict.update(self.latest_sampled_idxs_per_lb_task)
        return out_dict

    def resample_requests_lb(self, train: bool,
                             sampled_requests_per_task: Optional[int] = None,
                             task_batch_size: Optional[int] = None,
                             ) -> None:
        if train:
            tasks_subset = self.lb_training_tasks
        else:
            tasks_subset = self.lb_test_tasks

        if tasks_subset is not None:
            num_tasks = len(tasks_subset)
            self.latest_lb_tasks_names = tasks_subset
        else:
            self.latest_lb_tasks_names = self.lb_tasks
            num_tasks = self.num_lb_tasks

        if task_batch_size is not None and num_tasks > 0:
            tasks_idxs = np.random.choice(num_tasks, replace=False,
                                          size=task_batch_size)
            self.latest_lb_tasks_names = [self.latest_lb_tasks_names[i]
                                          for i in tasks_idxs]

        tasks_names = self.latest_lb_tasks_names

        sampled_idxs_per_lb_task = {}
        for task_n in tasks_names:
            num_task_prompts = self.num_prompts_per_lb_task[task_n]
            if sampled_requests_per_task is not None:
                sampled_idxs = np.random.choice(
                    num_task_prompts, replace=False,
                    size=sampled_requests_per_task)
            else:
                sampled_idxs = np.arange(num_task_prompts)
            sampled_idxs_per_lb_task[task_n] = sampled_idxs
        self.latest_sampled_idxs_per_lb_task = sampled_idxs_per_lb_task

    def evaluate(
        self,
        lm,
        train: bool,
        evolved_model: bool,
        pop_reps: int = 1,
        pop_idxs: Optional[np.array] = None,
        resample_requests: bool = True,
        sampled_requests_per_task: Optional[int] = None,
        task_batch_size: Optional[int] = None,
        limit: Optional[int] = None,
        replicate_requests: Optional[int] = None,
        build_chat_interface: bool = False,
        performance_per_request: bool = False,
        cache_param_stats_per_task: bool = False,
        model_kwargs: dict = {},
    ):

        out_dicts = [{} for _ in range(pop_reps)]
        if train:
            tasks_subset = self.lb_training_tasks
        else:
            tasks_subset = self.lb_test_tasks
        if len(tasks_subset) > 0:
            lb_dicts, lb_stats = self.evaluate_lb_tasks_for_pop(
                lm=lm,
                pop_reps=pop_reps, pop_idxs=pop_idxs,
                resample_requests=resample_requests,
                sampled_requests_per_task=sampled_requests_per_task,
                tasks_subset=tasks_subset, task_batch_size=task_batch_size,
                limit=limit, build_chat_interface=build_chat_interface,
                performance_per_request=performance_per_request,
                cache_param_stats_per_task=cache_param_stats_per_task,
                model_kwargs=model_kwargs)

            out_dicts = merge_list_of_dicts(out_dicts, lb_dicts)
            out_dicts = merge_list_of_dicts(out_dicts, lb_stats)

        return out_dicts

    def get_latest_sampled_idxs(self, train=True):
        lb_tasks_names = self.latest_lb_tasks_names
        if train:
            tasks_subset = self.lb_training_tasks
        else:
            tasks_subset = self.lb_test_tasks
        all_idxs = {}
        if lb_tasks_names is not None:
            lb_tasks_names = [t_n for t_n in lb_tasks_names
                              if t_n in tasks_subset]
            for task_n in lb_tasks_names:
                sampled_idxs = self.latest_sampled_idxs_per_lb_task[task_n]
                if sampled_idxs is None:
                    all_idxs[task_n] = np.arange(
                        self.num_prompts_per_lb_task[task_n])
                else:
                    all_idxs[task_n] = sampled_idxs

        return all_idxs

    def evaluate_lb_tasks_for_pop(
        self,
        lm,
        pop_reps: int,
        pop_idxs: Optional[np.array] = None,
        resample_requests: bool = True,
        sampled_requests_per_task: Optional[int] = None,
        tasks_subset: Optional[list] = None,
        task_batch_size: Optional[int] = None,
        # only used for debugging in the absence of sampled_requests_per_task
        limit: Optional[int] = None,
        use_cached_kv_if_available: bool = True,
        build_chat_interface: bool = False,
        performance_per_request: bool = False,
        cache_param_stats_per_task: bool = False,
        model_kwargs: dict = {},
    ):

        stats = [{} for _ in range(pop_reps)]

        if resample_requests:
            if tasks_subset is not None:
                num_tasks = len(tasks_subset)
                self.latest_lb_tasks_names = tasks_subset
            else:
                self.latest_lb_tasks_names = self.lb_tasks
                num_tasks = self.num_lb_tasks

            if task_batch_size is not None:
                tasks_idxs = np.random.choice(num_tasks, replace=False,
                                              size=task_batch_size)
                self.latest_lb_tasks_names = [self.latest_lb_tasks_names[i]
                                              for i in tasks_idxs]

        model_kwargs = dict(pop_reps=pop_reps, pop_idxs=pop_idxs,
                            model_kwargs=model_kwargs)

        tasks_names = self.latest_lb_tasks_names

        sampled_idxs_per_lb_task = {}
        sampled_task_prompts = {}
        sampled_task_jsons = {}
        for task_n in tasks_names:
            task_prompts = self.lb_prompts_per_task[task_n]
            task_jsons = self.lb_jsons_per_task[task_n]
            if not resample_requests:
                sampled_idxs = self.latest_sampled_idxs_per_lb_task[task_n]
                prompts = [task_prompts[i] for i in sampled_idxs]
                jsons = [task_jsons[i] for i in sampled_idxs]

            elif sampled_requests_per_task is not None:
                sampled_idxs = np.random.choice(
                    len(task_prompts), replace=False,
                    size=sampled_requests_per_task)
                prompts = [task_prompts[i] for i in sampled_idxs]
                jsons = [task_jsons[i] for i in sampled_idxs]
            else:
                sampled_idxs = None
                if limit is not None:
                    prompts = task_prompts[:limit]
                    jsons = task_jsons[:limit]
                else:
                    prompts = task_prompts
                    jsons = task_jsons
            sampled_idxs_per_lb_task[task_n] = sampled_idxs

            sampled_task_jsons[task_n] = jsons
            sampled_task_prompts[task_n] = prompts

        self.latest_sampled_idxs_per_lb_task = sampled_idxs_per_lb_task

        resps_per_task = {}
        pop_task_scores = [{} for _ in range(pop_reps)]
        if performance_per_request:
            for pop_i in range(pop_reps):
                stats[pop_i]['performance_per_request'] = {}
        for task_n, prompts in sampled_task_prompts.items():
            if (self.prefetched_task_tensors[task_n] is not None
                    and use_cached_kv_if_available):
                raise NotImplementedError

            build_chat_interface_for_task = False
            if build_chat_interface:
                dataset_n = task_n.split('/')[1]
                if dataset_n not in ["trec", "triviaqa", "samsum", "lsht",
                                     "lcc", "repobench-p"]:
                    build_chat_interface_for_task = True

            all_classes = None

            jsons = sampled_task_jsons[task_n]

            task_kwargs = dict(
                max_gen_tokens=self.lb_task2maxlen[task_n],
                stop_gen=self.lb_taskstopgen[task_n],
                build_chat_interface=build_chat_interface_for_task)

            task_outputs = lm.evaluate_lb(dataset_samples=prompts,
                                          **task_kwargs, **model_kwargs)
            task_outputs = task_outputs
            n_task_outputs = len(task_outputs)
            n_outputs_per_pop_idx = n_task_outputs // pop_reps

            assert n_outputs_per_pop_idx == len(jsons)

            task_ouputs_per_pop_idx = [
                task_outputs[i:i + n_outputs_per_pop_idx]
                for i in range(0, n_task_outputs, n_outputs_per_pop_idx)]
            dicts_to_store = []
            has_length = False

            for j in range(pop_reps):
                prediction_list, answers_list, length_list = [], [], []
                for i, (json_obj, prompt) in enumerate(zip(jsons, prompts)):
                    all_classes = json_obj["all_classes"]
                    answers = json_obj["answers"]
                    if "length" in json_obj:
                        length = json_obj["length"]
                        has_length = True
                    else:
                        length = -1
                        assert not has_length

                    pred = task_ouputs_per_pop_idx[j][i]
                    prediction_list.append(pred)
                    answers_list.append(answers)
                    length_list.append(length)

                    if self.store_gen_outputs:
                        prompt_dict = dict(
                            pred=pred,
                            answers=answers,
                            all_classes=all_classes,
                            length=length,
                        )
                        if pop_idxs is not None:
                            prompt_dict['pop_idx'] = pop_idxs[j]
                        dicts_to_store.append(prompt_dict)

                score, all_scores = get_score(
                    task=task_n[task_n.index('/') + 1:],  # strip task prefix
                    predictions=prediction_list,
                    answers=answers_list,
                    all_classes=all_classes)

                pop_task_scores[j][task_n] = score
                if performance_per_request:
                    if sampled_idxs_per_lb_task[task_n] is None:
                        if limit is None:
                            sampled_prompt_idxs = list(range(len(all_scores)))
                        else:
                            sampled_prompt_idxs = list(range(limit))
                    else:
                        sampled_prompt_idxs = sampled_idxs_per_lb_task[task_n]
                    assert (len(sampled_prompt_idxs) ==
                            len(all_scores))
                    stats[j]['performance_per_request'][task_n] = {
                        prompt_idx: prompt_score for prompt_idx, prompt_score in
                        zip(sampled_prompt_idxs, all_scores)}
            if cache_param_stats_per_task:
                memory_policy_stats = lm.model.get_param_stats()
                for k, v in memory_policy_stats.items():
                    self.cached_per_task_stats[
                        f'{task_n[task_n.index("/") + 1:]}/' + k] = v
            if self.store_gen_outputs:
                pass

        return pop_task_scores, stats
