import numpy as np
import torch
import debugpy
# DistributedSampler 
from torch.utils.data import DistributedSampler
# if rank == 0: import my_tools; my_tools.debug()
def debug(port=5678):
    debugpy.listen(port)
    print(f"Waiting for debugger attach, port: {port}")
    debugpy.wait_for_client()
    debugpy.breakpoint()

def format_params(num_params):
    """
    将参数数量转换为带单位的字符串
    :param num_params: 参数数量
    :return: 带单位的字符串
    """
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)

def get_model_params(model, parent_name=''):
    params_dict = {}
    total_params = sum(p.numel() for p in model.parameters())
    if parent_name == '':
        parent_name = 'all'
        
    params_dict[parent_name] = format_params(total_params)

    for name, child in model.named_children():
        child_name = name
        params_dict[parent_name+'.'+name] = get_model_params(child, child_name)

    return params_dict

    
def occ_mem(device):
    # get idle memory
    free_mem = torch.cuda.mem_get_info(device=device)[0]
    chunk_shape = (100, 1024, 1024)
    chunk_size = 4 * chunk_shape[0] * chunk_shape[1] * chunk_shape[2]
    res = []
    for _ in range(free_mem//chunk_size):
        res.append(torch.ones(chunk_shape, device=device))
        size = res[-1].element_size() * res[-1].nelement()
    for t in res:
        del t


def custom_repr(self):
    def get_first(j):
        _value = self
        for i in range(len(self.shape)-1):
            _value = _value[0]
        return _value[j]
        
    if self.shape[-1] == 0:
        stri = ''
    elif self.shape[-1] == 1:
        stri = '[' + str(get_first(0)) + ']'
    else:
        stri = '['*len(self.shape) + f'{get_first(0)}, {get_first(1)}' + '...'
    return f"{self.shape} array: {stri}"  
np.set_string_function(custom_repr, repr=True)

original_repr = torch.Tensor.__repr__
def custom_repr2(self):
    shape_str = str(tuple(self.shape))
    if shape_str[-2] == ',':
        shape_str = shape_str[:-2] + ')'
    dtype = str(self.dtype).split('torch.')[1]
    return f"{tuple(self.shape)} '{dtype}' '{self.device}' {original_repr(self)}"
torch.Tensor.__repr__ = custom_repr2

import inspect
def where_this(x):
    '''
    x: object or class, find where it is defined
    '''
    if inspect.isclass(x):
        return (inspect.getfile(x), x.__name__)
    else:
        return (inspect.getfile(inspect.getmodule(x)), x.__class__.__name__)