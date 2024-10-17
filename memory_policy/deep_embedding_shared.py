import os
import pdb
import copy
import math
import numbers
import abc
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Callable, List

from omegaconf import OmegaConf, DictConfig
import hydra

import torch
from torch import nn
import torch.utils.checkpoint


def convert_to_tensor(
        el: Union[List[float], np.ndarray, torch.Tensor],
        ) -> torch.Tensor:
    if isinstance(el, torch.Tensor):
        return el
    else:
        el = torch.tensor(el)
    return el

def cos_sin_seq_embeddings(
        length,
        embed_dim,
        max_freq=10000,
        
        
        ):
    assert embed_dim % 2 == 0, 'Embedding dimension should be even '
    positions = np.arange(length)
    embed_dim_per_op = embed_dim // 2
    
    
    freq_coeff = np.arange(embed_dim_per_op, dtype=np.float64)/embed_dim_per_op
    freq_coeff = 1/(max_freq**freq_coeff) 
    
    freq_values = np.expand_dims(positions, axis=-1)*freq_coeff
    embeddings = np.concatenate(
        [np.sin(freq_values), np.cos(freq_values)], axis=-1)
    return embeddings  

class Embedding(nn.Module, abc.ABC):
    def __init__(self, embed_dim):
        nn.Module.__init__(self=self,)
        self.embed_dim = embed_dim
        if embed_dim is not None:
            self.set_embed_dim(embed_dim=embed_dim)

    def set_embed_dim(self, embed_dim):
        self.embed_dim: int = embed_dim
        assert self.embed_dim is not None, 'make sure embed_dim is not None'

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PositionalEmbedding(Embedding):
    def __init__(self, max_position_id, embed_dim, max_freq):
        self.max_position_id = max_position_id
        self.max_freq = max_freq
        Embedding.__init__(self=self, embed_dim=embed_dim)


    def set_embed_dim(self, embed_dim):
        Embedding.set_embed_dim(self=self, embed_dim=embed_dim)
        embeddings = cos_sin_seq_embeddings(
            
            
            length=self.max_position_id + 1,
            embed_dim=embed_dim,
            max_freq=self.max_freq,
            ).astype('float32')
        self.register_buffer('embeddings', torch.tensor(embeddings))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        
        
        
        
        emb =  self.embeddings[x] 
        
        return emb

