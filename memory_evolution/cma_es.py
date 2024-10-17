import typing as tp
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from memory_evolution.base import MemoryEvolution
from omegaconf import OmegaConf, DictConfig
import hydra
import numpy as np


def full_eigen_decomp(
    C: torch.Tensor, gen_counter: torch.Tensor,
    shift_first_gen: bool = True,
) -> torch.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform eigendecomposition of covariance matrix."""
    # small shift for first iter.
    if shift_first_gen:
        C = C + 1e-6 * (gen_counter == 0).to(dtype=C.dtype)
    # Force matrix to be symmetric - account for num. errors
    C = (C + C.T) / 2
    D_squared, B = torch.linalg.eigh(C, UPLO= 'L')
    D = torch.sqrt(torch.where(D_squared <= 0, 1e-20, D_squared))
    C = B@torch.diag_embed(D**2)@B.T
    # returns 'reconstructed' C mx 
    return C, B, D

def get_cma_defaults(
    pop_size: int,
    elite_pop_size: int,
    param_size: int,
) -> tp.Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Utility helper to create truncated elite weights for mean
    update and full weights for covariance update."""

    # clipped for numerical stability and avoiding overflow
    clipped_param_size = min(param_size, 40000)
    
    # default weights, neg. log scaling with half nedagtive weights (closer to 
    # 0) 
    weights_prime = torch.tensor(
        [np.log((pop_size + 1) / 2) - np.log(i + 1) for i in range(pop_size)],
        dtype=torch.float32
    )

    # effective elite size
    mu_eff = (torch.sum(weights_prime[:elite_pop_size]) ** 2) / torch.sum(
        weights_prime[:elite_pop_size] ** 2
    )

    # effective pop size for weights beyond the truncated idx
    mu_eff_minus = (torch.sum(weights_prime[elite_pop_size:])**2)/torch.sum(
        weights_prime[elite_pop_size:] ** 2)

    # lrates for rank-one and rank-μ C updates
    alpha_cov = 2
    c_1 = alpha_cov / ((clipped_param_size + 1.3)**2 + mu_eff)

    c_mu = torch.minimum(
        1 - c_1 - 1e-8,
        alpha_cov * (mu_eff - 2 + 1/mu_eff) /
        ((clipped_param_size + 2)**2 + alpha_cov*mu_eff/2),
    )

    min_alpha = torch.minimum(
        1 + c_1 / c_mu, # default value to make total weights close to 0
        1 + (2*mu_eff_minus) / (mu_eff + 2), 
        )

    min_alpha = torch.minimum(
        min_alpha,
        (1 - c_1 - c_mu) / (param_size * c_mu), # bound for pos. def.
        )
    
    positive_sum = torch.sum(
        weights_prime*(weights_prime > 0).to(dtype=weights_prime.dtype))
    negative_sum = torch.sum(torch.abs(
        weights_prime *(weights_prime < 0).to(dtype=weights_prime.dtype)))
    
    weights = torch.where(
        weights_prime >= 0,
        # pos weights sum to one
        1 / positive_sum * weights_prime,
        # neg. weights to make decay of prev. C ~ 0 (bar stability clipping)
        min_alpha / negative_sum * weights_prime,
    )

    # weights truncated only includes weights for the elite pop size top members
    weights_truncated = weights.clone()
    weights_truncated[elite_pop_size:] = 0
    return weights, weights_truncated, mu_eff, c_1, c_mu



class CMA_ES(MemoryEvolution):
    '''Non-memory dependant queries, based on evojax's implementation:
       https://github.com/google/evojax/blob/main/evojax/algo/pgpe.py'''
    def __init__(
            self,
            # Memory evolution params
            # min. recomended pop size 4 + floor(3 ln n) - 17 for 100 params
            pop_size: int, # equal to n_replicas (of dividing)
            param_size: int, 
            param_clip: tp.Optional[float] = None,
            clip_param_min: tp.Optional[float] = None,
            clip_param_max: tp.Optional[float] = None,
            score_processing: tp.Optional[str] = None,

            # CMA-ES params see https://arxiv.org/pdf/1604.00772 page 28
            elite_ratio: float = 0.5,
            # Optional params can be computed based the default recomended
            # values abovev (pg. 29 and pg.31)
            # mu_eff: float,
            c_1: tp.Optional[float] = None,
            c_mu: tp.Optional[float] = None,
            c_sigma: tp.Optional[float] = None,
            d_sigma: tp.Optional[float] = None,
            c_c: tp.Optional[float] = None,
            # expected_normal_dist: tp.Optional[float] = None,
            c_m: tp.Optional[float] = 1.0,
            init_sigma: tp.Optional[float] = 0.065,
            init_param_range: tp.Optional[tp.Tuple[float, float]] = None,
            prefer_mean_to_best: bool = False,
            shift_first_gen: bool = True,
        ):
        super().__init__(pop_size=pop_size,
                         param_size=param_size, 
                         param_clip=param_clip,
                         clip_param_min=clip_param_min,
                         clip_param_max=clip_param_max,
                         score_processing=score_processing,
                         prefer_mean_to_best=prefer_mean_to_best,
                         )

        # clipped for numerical stability and avoiding overflow
        self.clipped_param_size = min(param_size, 40000)

        print(f'# optimized parameters: {param_size}')
        self.init_cma_es_params(
            elite_ratio=elite_ratio,
            c_1=c_1,
            c_mu=c_mu,
            c_sigma=c_sigma,
            d_sigma=d_sigma,
            c_c=c_c,
            c_m=c_m,
            init_sigma=init_sigma,
        )
        
        self.init_param_range = init_param_range
        self.shift_first_gen = shift_first_gen
        mean = torch.zeros(self.param_size)
        if init_param_range is not None:
            mean.uniform_(from_=init_param_range[0], to=init_param_range[1])
        best_member = mean.clone()
        best_fitness = torch.zeros([]) - torch.finfo(best_member.dtype).max

        g = torch.zeros([])
        evo_path_sigma = torch.zeros(self.param_size)
        evo_path_c = torch.zeros(self.param_size)
        sigma = torch.tensor(init_sigma)
        C = torch.eye(self.param_size)
        
        # still save buffers as persistent state to allow resuming training
        self.g = nn.Parameter(data=g, requires_grad=False) # gen. number
        self.best_member = nn.Parameter(data=best_member, requires_grad=False)
        self.best_fitness = nn.Parameter(data=best_fitness, requires_grad=False)
        self.evo_path_c = nn.Parameter(data=evo_path_c, requires_grad=False)
        self.evo_path_sigma = nn.Parameter(data=evo_path_sigma, 
                             requires_grad=False)

        self.mean = nn.Parameter(data=mean, requires_grad=False)
        self.sigma = nn.Parameter(data=sigma, requires_grad=False)
        self.C = nn.Parameter(data=C, requires_grad=False)

        sample_C, sample_B, sample_D = full_eigen_decomp(
            C=self.C, gen_counter=self.g, shift_first_gen=self.shift_first_gen)

        self.sample_C = nn.Parameter(data=sample_C, requires_grad=False)
        self.sample_B = nn.Parameter(data=sample_B, requires_grad=False)
        self.sample_D = nn.Parameter(data=sample_D, requires_grad=False)

        x, y = self.sample_pop(
            num_samples=self.pop_size,
            mean=self.mean,
            sigma=self.sigma,
            D=self.sample_D,
            B=self.sample_B,
            )

        self.x = nn.Parameter(data=x, requires_grad=False)
        self.y = nn.Parameter(data=y, requires_grad=False)

        

    def init_cma_es_params(
            self,
            elite_ratio: float,
            c_1: tp.Optional[float] = None,
            c_mu: tp.Optional[float] = None,
            c_sigma: tp.Optional[float] = None,
            d_sigma: tp.Optional[float] = None,
            c_c: tp.Optional[float] = None,
            c_m: float = 1.0,
            init_sigma: float = 0.065,
            ):
        assert 0 <= elite_ratio <= 1
        self.elite_ratio = elite_ratio
        self.elite_pop_size = max(1, int(self.pop_size * self.elite_ratio))
        self.strategy_name = "CMA_ES"

        # Set core kwargs es_params
        self.init_sigma = init_sigma

        self.c_1 = c_1
        self.c_mu = c_mu
        self.c_sigma = c_sigma
        self.d_sigma = d_sigma
        self.c_c = c_c
        self.c_m = c_m

        weights, weights_truncated, mu_eff, c_1, c_mu = get_cma_defaults(
                pop_size=self.pop_size,
                elite_pop_size=self.elite_pop_size,
                param_size=self.param_size,)

        self.mu_eff = nn.Parameter(mu_eff, requires_grad=False)
        self.weights = nn.Parameter(weights, requires_grad=False)
        self.weights_truncated = nn.Parameter(weights_truncated,
                                              requires_grad=False)

        # step size evo path lr
        c_sigma = (self.mu_eff + 2)/(self.param_size + self.mu_eff + 5)
        # step size damping
        d_sigma = 1 + 2*torch.clamp_min(torch.sqrt(
            (self.mu_eff - 1)/(self.param_size + 1)) - 1, min=0) + c_sigma
        
        # rank 1 evo path lr
        c_c = (4 + self.mu_eff/self.param_size)/(
            self.param_size + 4 + 2*self.mu_eff / self.param_size)
        
        # override where needed
        if self.c_1 is None:
            self.c_1 = c_1
        if self.c_mu is None:
            self.c_mu = c_mu
        if self.c_sigma is None:
            self.c_sigma = c_sigma
        if self.d_sigma is None:
            self.d_sigma = d_sigma
        if self.c_c is None:
            self.c_c = c_c
        if self.c_m is None:
            self.c_m = c_m
        
        # clipped for numerical stability and avoiding overflow
        clipped_param_size = min(self.param_size, 40000)
        # chi dist. used for step size control and evo. path clipping
        self.expected_normal_mag = np.sqrt(self.param_size)*(
            1 - (1/(4*self.param_size)) + 1/(21*(clipped_param_size**2)))

    def sample_pop(
            self,
            num_samples,
            mean,
            sigma,
            D,
            B,
    ) -> torch.Tensor:
        z = torch.randn([num_samples, self.param_size], 
                        device=D.device)
        
        y = z@torch.diag_embed(D) @ B.T

        # samples
        x = mean + sigma*y
        return x, y
    
    def ask(self,) -> torch.Tensor:
        """Ask the algorithm for a population of parameters."""

        C, B, D = full_eigen_decomp(C=self.C, gen_counter=self.g,
                                    shift_first_gen=self.shift_first_gen)

        self.sample_C.data.copy_(C)
        self.sample_B.data.copy_(B)
        self.sample_D.data.copy_(D)

        x, y = self.sample_pop(
            num_samples=self.pop_size,
            mean=self.mean,
            sigma=self.sigma,
            D=self.sample_D,
            B=self.sample_B,
            )

        self.x.data.copy_(x)
        self.y.data.copy_(y)

        return x
    
    def sample_candidates(
        self,
        num_candidates: int,
        temperature: float = 1.0,
        ) -> torch.Tensor:

        C, B, D = full_eigen_decomp(C=self.C, gen_counter=self.g, 
                                    shift_first_gen=self.shift_first_gen)

        x, y = self.sample_pop(
            num_samples=num_candidates,
            mean=self.mean,
            sigma=self.sigma*temperature,
            D=D,
            B=B,
            )

        return x


    def tell(self, fitness) -> None:
        """Report the fitness of the population to the algorithm."""
        
        self.g.data.add_(torch.ones_like(self.g))
        # best pop idxs (highest fitness) first, to match the weights
        sorted_fitness, sorted_pop_idxs = torch.sort(fitness, descending=True)
        
        sorted_x = self.x[sorted_pop_idxs]
        sorted_y = self.y[sorted_pop_idxs]

        
        # summing elite deviations y_w
        sum_elite_y = (sorted_y*self.weights_truncated.unsqueeze(-1)).sum(dim=0)

        # update mean inplace
        self.mean.data.add_(self.c_m*self.sigma*sum_elite_y)

        nroot_C = self.sample_B@torch.diag_embed(
            1/self.sample_D)@self.sample_B.T

        norm_sum_elite_y = (nroot_C*sum_elite_y).sum(-1)

        # step size evolution path update
        self.evo_path_sigma.data.copy_(
            (1 - self.c_sigma)*self.evo_path_sigma + torch.sqrt(
                self.c_sigma*(2 - self.c_sigma)*self.mu_eff)*norm_sum_elite_y)


        # current evo path magnitude
        norm_evo_path_sigma = torch.norm(self.evo_path_sigma)

        # check edge case too m=large magnitude of evo. path relative to actual
        # update (in which case simply shrink current evo path)
        h_sigma_cond_left = norm_evo_path_sigma/torch.sqrt(
            1 - torch.pow(1 - self.c_sigma, 2*self.g))
        
        h_sigma_cond_right = (
            1.4 + 2/(self.param_size + 1))*self.expected_normal_mag
        # 0 whenever edge case is True
        h_sigma = (h_sigma_cond_left < h_sigma_cond_right).to(
            dtype=self.evo_path_c.data.dtype)
        
        # rank-1 evolution path update
        self.evo_path_c.data.copy_(
            (1 - self.c_c)*self.evo_path_c + h_sigma*torch.sqrt(
                self.c_c*(2 - self.c_c)*self.mu_eff)*sum_elite_y)


        # adjusted weights 
        w_io = self.weights*torch.where(self.weights > 0, 1,
            self.param_size/(torch.norm(nroot_C@sorted_y.T, dim=0)**2 + 1e-20))
         
        # correcting update when h is 0
        delta_h_sigma = (1 - h_sigma)*self.c_c*(2 - self.c_c)
        rank_one = torch.outer(self.evo_path_c, self.evo_path_c)
        
        # outer prod for each element
        rank_mu = torch.einsum('i,ij,ik->kj', w_io, sorted_y, sorted_y)
        
        # covariance update
        self.C.data.copy_(
            (1 + self.c_1*delta_h_sigma - self.c_1 - self.c_mu*torch.sum(
                self.weights))*self.C + self.c_1*rank_one + self.c_mu*rank_mu)
        
        # update step size based on ratio with exp. evo path length for indep.
        # updates
        self.sigma.data.copy_(self.sigma*torch.exp((self.c_sigma/self.d_sigma)*(
            norm_evo_path_sigma/self.expected_normal_mag - 1)))

        improvement = self.best_fitness < sorted_fitness[0]
        self.best_member.data.copy_(torch.where(
            improvement, sorted_x[0], self.best_member))

        self.best_fitness.data.copy_(torch.where(
            improvement, sorted_fitness[0], self.best_fitness))
        
        if (sorted_fitness[0] > self.best_fitness).cpu().item():
            self.store_buffers(
                buffers=self.get_stored_buffers(best=False),
                best=True,
                )


    @property
    def best_params(self,) -> torch.Tensor:
        if self.prefer_mean_to_best:
            return self.mean
        else:
            return self.best_member
    
    def store_best_params(self, x, fitness=None):
        self.best_member.data.copy_(x)
        if fitness is not None:
            self.best_fitness.data.copy_(fitness)
    
    def load_init(self, init_param):
        '''Loads custom initialization values'''
        if init_param is not None:
            self.mean.data.copy_(init_param.to(dtype=self.mean.dtype,
                                            device=self.mean.device))
            if self.g.item() == 0:
                self.best_member.data.copy_(self.mean)
    
    def get_stats(self,):
        '''Returns statistics for logging'''
        stats = {
            'evo_stats/sample_D_mean': self.sample_D.mean().item(),
            'evo_stats/sample_D_std': self.sample_D.std().item(),
            'evo_stats/sample_D_min': self.sample_D.min().item(),
            'evo_stats/sample_D_max': self.sample_D.max().item(),
            'evo_stats/mean_mean': self.mean.mean().item(),
            'evo_stats/mean_std': self.mean.std().item(),
            'evo_stats/mean_min': self.mean.min().item(),
            'evo_stats/mean_max': self.mean.max().item(),
            'evo_stats/step_size': self.sigma.mean().item(),
            }
        return stats