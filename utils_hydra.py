from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from torch.nn import functional as F
import torch
import transformers
import numpy as np


from hydra import compose, initialize
from main import make_eval_model, make_task_sampler, wandb_init
import omegaconf
import hydra
import time


# File containing imports to be used in hydra configs/for instantiating hydra
# objects outside main

def initialize_cfg(
        config_path="cfgs",
        hydra_overrides: dict = {},
        log_yaml: bool = False,

):
    with initialize(version_base=None, config_path=config_path,
                     job_name="test_app"):
        cfg = compose(config_name="config",
                      overrides=hydra_overrides,
                      )
        if log_yaml:
            print('Loading the following configurations:')
            print(omegaconf.OmegaConf.to_yaml(cfg))
    return cfg


def load_run_cfgs_trainer(
    run_file_path_location: str,
    hydra_overrides_from_dict: dict = dict(
        wandb_log="false",
        wandb_project="scratch",
    ),
    task_sampler_kwargs: dict = dict(
        tasks=["lb/passage_retrieval_en"],
    ),
    config_path="cfgs",
    batch_size: int = 1,
):
    print(f"Loading model specified in {run_file_path_location}")
    start_time = time.time()
    hydra_overrides = [
        f"run@_global_={run_file_path_location}",
    ]

    hydra_overrides_from_dict = [f"{k}={v}" for k, v in
                                 hydra_overrides_from_dict.items()]

    hydra_overrides = hydra_overrides + hydra_overrides_from_dict

    cfg = initialize_cfg(
        config_path=config_path,
        hydra_overrides=hydra_overrides,
    )

    (memory_policy, memory_model, memory_evaluator, evolution_algorithm,
     auxiliary_loss) = make_eval_model(cfg=cfg)

    task_sampler = make_task_sampler(cfg=cfg, **task_sampler_kwargs)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        evaluation_model=memory_evaluator,
        task_sampler=task_sampler,
        evolution_algorithm=evolution_algorithm,
        auxiliary_loss=auxiliary_loss,
    )
    params, buffers = trainer.sample_and_synchronize_params(best=True)
    memory_model = memory_model
    memory_model.set_memory_params(params=params)

    if memory_model.memory_policy_has_buffers_to_merge():
        memory_model.load_buffers_dict(buffers_dict=buffers)

    batch_idxs = np.zeros([batch_size])
    memory_policy.set_params_batch_idxs(batch_idxs)
    print("Time taken:", round(time.time() - start_time))
    return trainer  # contains all other models
