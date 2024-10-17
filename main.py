import os
import torch
import random
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
from omegaconf import DictConfig, OmegaConf
import hydra
import copy
import datetime


def ddp_setup(
        backend: str = "nccl",
        ddp_timeout_limit: str = '0:6:0',  # days:hours:minutes
):
    days, hours, seconds = ddp_timeout_limit.split(':')
    timeout_delta = datetime.timedelta(
        days=int(days), hours=int(hours), seconds=int(seconds))
    init_process_group(backend=backend, timeout=timeout_delta)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_dist_info():
    local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', -1))
    global_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))
    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '-1'))

    local_rank = int(os.getenv(
        'LOCAL_RANK', -1)) if local_rank == -1 else local_rank
    global_rank = int(os.getenv(
        'RANK', -1)) if global_rank == -1 else global_rank
    world_size = int(os.getenv(
        'WORLD_SIZE', -1)) if world_size == -1 else world_size

    if global_rank != -1:
        os.environ['RANK'] = str(global_rank)  # Needed by torch distributed.
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)

    return local_rank, global_rank, world_size


def wandb_init(cfg):
    import wandb
    config_dict = OmegaConf.to_container(
        # allow missing values for memory experiments
        cfg, resolve=True, throw_on_missing=False,
    )
    # wandb has a 128-size character limit on the group name
    wandb.init(
        project=cfg.wandb_config.wandb_project,
        group=cfg.wandb_config.wandb_group_name[:127],
        name=cfg.wandb_config.wandb_run_name[:127],
        config=config_dict,
    )
    return wandb


def make_eval_model(cfg, log_prefix='...'):
    print(log_prefix + 'Instantialting llm...')
    pretrained_llm = hydra.utils.call(cfg.pretrained_llm, _convert_="object")
    tokenizer = hydra.utils.call(cfg.tokenizer)

    print(log_prefix + 'Instantialting memory policy...')
    memory_policy = hydra.utils.instantiate(cfg.memory_policy,
                                            _convert_="object")

    print(log_prefix + 'Instantialting memory llm...')
    memory_model = hydra.utils.instantiate(
        cfg.memory_model, model=pretrained_llm, memory_policy=memory_policy,)

    print(log_prefix + 'Instantialting evaluation module...')
    memory_evaluator = hydra.utils.instantiate(
        cfg.memory_evaluator, model=memory_model, tokenizer=tokenizer)

    print(log_prefix + 'Instantialting evolution module...')
    evolution_algorithm = hydra.utils.instantiate(
        cfg.evolution_algorithm, param_size=memory_policy.param_size,
        _recursive_=False)

    init_param = memory_policy.get_init_param_values()

    evolution_algorithm.load_init(init_param=init_param)

    if cfg.auxiliary_loss is not None:
        auxiliary_loss = hydra.utils.instantiate(cfg.auxiliary_loss,
                                                 memory_policy=memory_policy)
    else:
        auxiliary_loss = None

    print(log_prefix + 'Finished instantiations.')
    return (memory_policy, memory_model, memory_evaluator, evolution_algorithm,
            auxiliary_loss)


def make_task_sampler(cfg, log_prefix='', **task_sampler_kwargs):
    print(log_prefix + f'Instantiating tasks: {cfg.task_sampler.tasks}; with '
          f' corresponding metrics: {cfg.task_sampler.metrics}')
    task_sampler = hydra.utils.instantiate(
        cfg.task_sampler, _convert_='none',
        **task_sampler_kwargs)
    return task_sampler


def stochasticity_setup(cfg, seed_offset=0, log_prefix=''):
    print(log_prefix + f'Global rank used for seed offset {seed_offset}')
    np.random.seed(cfg.seed + seed_offset)
    torch.manual_seed(cfg.seed + seed_offset)
    random.seed(cfg.seed + seed_offset)

    # NOTE: likely can remove offset
    if cfg.deterministic_behavior:
        print('WARNING: training with deterministic behavior')
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


@hydra.main(version_base=None, config_path='cfgs', config_name='config')
def main(cfg: DictConfig):
    _, global_rank, n_ddp = get_dist_info()
    is_ddp = global_rank > -1
    if is_ddp:
        ddp_setup(backend=cfg.backend, ddp_timeout_limit=cfg.ddp_timeout_limit)
        master_process = global_rank == 0
        seed_offset = global_rank
    else:
        master_process = True
        seed_offset = 0

    if master_process:
        print(f"SHARED Working directory: {os.getcwd()}")
        print(f"SHARED Output directory: " +
              f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    log_prefix = ''
    if is_ddp:
        log_prefix = f'RANK {global_rank} ({n_ddp} total): '

    stochasticity_setup(cfg=cfg, seed_offset=seed_offset,
                        log_prefix=log_prefix)

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator, evolution_algorithm,
            auxiliary_loss) = make_eval_model(cfg=cfg, log_prefix=log_prefix)

        task_sampler = make_task_sampler(cfg=cfg, log_prefix=log_prefix)

        trainer = hydra.utils.instantiate(
            cfg.trainer,
            evaluation_model=memory_evaluator,
            task_sampler=task_sampler,
            evolution_algorithm=evolution_algorithm,
            auxiliary_loss=auxiliary_loss,
        )

        if cfg.wandb_config.wandb_log and master_process:
            wandb_init(cfg=cfg)

        with torch.no_grad():
            trainer.train()

        if is_ddp:
            destroy_process_group()


if __name__ == "__main__":
    with torch.no_grad():
        main()
