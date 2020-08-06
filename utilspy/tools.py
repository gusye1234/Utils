import torch
from typing import Callable, Iterable

def set_seed(seed):
    '''
        fix Randomness
    '''
    import random
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)   
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)
            
def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result
    
def TO(*tensors, **kwargs):
    if kwargs.get("device"):
        device = torch.device(kwargs['device'])
    else:
        device = torch.device('cpu')
    results = []
    for tensor in tensors:
        results.append(tensor.to(device))
    return results

class timer:
    """
    Time context manager for code block
    """
    from time import time
    TAPE = [-1] # global time record
    @staticmethod
    def get():
        return timer.TAPE[-1]
    def __init__(self, tape=None):
        self.tape = tape or timer.TAPE
    def __enter__(self):
        self.start = timer.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tape.append(timer.time()-self.start)
        
def load(model, file):
    try:
        model.load_state_dict(torch.load(file))
    except RuntimeError:
        model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
    except FileNotFoundError:
        raise FileNotFoundError(f"{file} NOT exist!!!")
    
def goMulti(func: Callable,
            jobs: Iterable,
            cores=None) -> Iterable:
    from multiprocessing import Pool, cpu_count
    cores = cores or cpu_count()//2
    return Pool(cores).map(func, jobs)









