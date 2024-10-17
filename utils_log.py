import numpy as np


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from lm_eval.models.utils import Collator


def stat_fn(list_of_lists, index=None):
    if index is None:
        flat_list = np.array(
            [v for vs in list_of_lists for v in vs])
    else:
        mean = np.mean(list_of_lists[i])
    mean = np.mean(flat_list)
    std = np.std(flat_list)
    above_avg = np.mean(flat_list > mean)
    max_v = np.max(flat_list)
    min_v = np.min(flat_list)
    return mean, std, above_avg, max_v, min_v


def initialize_stat_objects_for(
        self,
        score_name,
        stats=['mean', 'std', 'above_mean', 'max', 'min'],
):
    for stat in stats:
        init_list = [[] for _ in range(self.num_memory_layers)]
        setattr(self,  f'{score_name}_{stat}', init_list)
        raise NotImplementedError


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
