
import itertools

def expand_grid(param_grid: dict) -> list[dict]:
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = list(itertools.product(*values))

    configs = []
    for combo in combos:
        cfg = {k: v for k, v in zip(keys, combo)}
        configs.append(cfg)
    return configs