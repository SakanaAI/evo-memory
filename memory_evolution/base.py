import typing as tp
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np


class MemoryEvolution(nn.Module, ABC):
    '''Strategy for parameterizing the queries is independent of memory
       and strategy for optimizing the queries'''

    def __init__(self,
                 pop_size,  # equal to n_replicas (of dividing)
                 param_size,
                 score_processing: tp.Optional[str] = None,
                 param_clip: tp.Optional[float] = None,
                 clip_param_min: tp.Optional[float] = None,
                 clip_param_max: tp.Optional[float] = None,
                 prefer_mean_to_best: bool = False,

                 ):
        nn.Module.__init__(self=self,)
        self.stored_buffers_to_save = nn.ParameterDict()
        self.best_stored_buffers_to_save = nn.ParameterDict()
        self.pop_size = pop_size
        self.param_size = param_size
        self.clip_param_min = clip_param_min
        self.clip_param_max = clip_param_max

        if clip_param_min is None and param_clip is not None:
            self.clip_param_min = -param_clip
        if clip_param_max is None and param_clip is not None:
            self.clip_param_max = param_clip

        if isinstance(score_processing, str):
            score_processing = score_processing.lower()
            assert score_processing in ['none', 'rank']
            if score_processing == 'rank':
                scores_values = torch.arange(pop_size)/(pop_size - 1) - 0.5
                self.register_buffer(
                    name='scores_values',
                    tensor=scores_values,
                    persistent=False,
                )

        self.score_processing = score_processing

        self.prefer_mean_to_best = prefer_mean_to_best

    @abstractmethod
    def ask(self,) -> torch.Tensor:
        """Ask the algorithm for a population of parameters and update internal
           state."""
        raise NotImplementedError()

    @abstractmethod
    def tell(self, fitness) -> None:
        """Report the fitness of the population to the algorithm."""
        raise NotImplementedError()

    @property
    def best_params(self) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def best_buffer(self) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def sample_candidates(
        self,
        num_candidates: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Ask the algorithm for a population of parameters, temperature should
           optionally indicate how much the parameters should differ from the
           best/mean solution, depending on the specific algorithm."""
        raise NotImplementedError()

    def process_scores(self, fitness) -> torch.Tensor:
        """Preprocess the scores"""
        # fitness should be a pop_size-dim. vector
        if self.score_processing is None:
            return fitness
        elif self.score_processing == 'none':
            return fitness
        elif self.score_processing == 'rank':
            sorted_idxs = fitness.argsort(-1).argsort(-1)  # from min to max
            # from -0.5 to 0.5
            scores = self.scores_values[sorted_idxs]
            return scores
        else:
            raise NotImplementedError

    def store_best_params(self, x, fitness=None):
        raise NotImplementedError

    def store_best_buffers(self, buffers):
        self.store_buffers(buffers=buffers, best=True)

    def forward(self,
                ):
        '''Computes queries'''
        return self.ask()

    def get_stats(self,
                  ):
        '''Returns statistics for logging'''
        return {}

    def load_init(self, init_param):
        '''Loads custom initialization values'''
        pass

    def store_buffers(
            self,
            buffers: tp.Dict[str, torch.Tensor] = None,
            best: bool = False,
    ):
        if best:
            self.best_stored_buffers_to_save.update(
                {n: nn.Parameter(v, requires_grad=False)
                 for n, v in buffers.items()})
        else:
            self.stored_buffers_to_save.update(
                {n: nn.Parameter(v, requires_grad=False)
                 for n, v in buffers.items()})
            # buffers)

    def get_stored_buffers(self, best=False):
        if best and (not self.prefer_mean_to_best):
            tensor_dict = {k: v.data.clone() for k, v
                           in self.best_stored_buffers_to_save.items()}
        else:
            tensor_dict = {k: v.data.clone() for k, v
                           in self.stored_buffers_to_save.items()}
        return tensor_dict


class DummyEvolution(MemoryEvolution):
    '''To be returned if no params to optimize, i.e., param_size=0'''

    def __init__(self,
                 pop_size,
                 param_size,
                 param_clip,
                 score_processing):
        nn.Module.__init__(self,)
        assert param_size == 0
        self.pop_size = pop_size
        self.param_size = param_size
        self.param_clip = param_clip
        self.register_buffer(
            name='dummy_params', tensor=torch.zeros(
                [self.pop_size, self.param_size],), persistent=False,)

    def ask(self) -> torch.Tensor:
        return self.dummy_params

    def tell(self, fitness) -> None:
        """Report the fitness of the population to the algorithm."""
        pass

    @property
    def best_params(self) -> torch.Tensor:
        return self.dummy_params[0]

    def forward(self,
                ):
        '''Computes queries'''
        return self.ask()

    def sample_candidates(
        self,
        num_candidates: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        pass
