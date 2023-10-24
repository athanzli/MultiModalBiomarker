
import numpy as np
import torch
import ctypes
from typing import Union, List, Optional
import pandas as pd

def tensor2numpy(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    np_array = np.empty(tensor.size(), dtype=np.float32)
    tensor_ptr = tensor.data_ptr()
    np_array_ptr = np_array.ctypes.data
    size = tensor.numel() * tensor.element_size()
    ctypes.memmove(np_array_ptr, tensor_ptr, size)
    return np_array

def convert2tensor(
        x: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        dtype = torch.float32):
    if isinstance(x, torch.Tensor):
        x = x.to(dtype)
    else:
        x = np.array(x)
        x = torch.tensor(x, dtype=dtype)
    return x

