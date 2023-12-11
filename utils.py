
import numpy as np
import torch
import ctypes
from typing import Union, List, Optional, Tuple
import pandas as pd
import numpy as np
import random
import os
import captum
import matplotlib.pyplot as plt

def project_matrix_onto_subspace(A, M):
    P = A @ M.T @ M
    return P

def effective_attention(A, V):
    r""" Inspired by paper: Effective Attention Sheds Light On Interpretability
    
    Shape:
        - A: (n_tokens, n_tokens_cond)
        - V: (n_tokens, d_v)
    """
    U, S, _ = torch.linalg.svd(V, full_matrices=True) # need full SVD for left null space
    # rows of U that correspond to zero singular values (u_1, ..., u_k) 
    #   are bases of the left null space of V
    cols = torch.arange(S.shape[0], U.shape[0])
    LN_V = U[:, cols].T # LN_V has the rows of U.T that correspond to zero singular values. LN_V is the left null space of V
    # projection of (rows of) A onto the left null space of V.
    #   projection of a matrix A onto a subspace spanned by orthonormal set {u_1, u_2, ..., u_k}
    #   This projection matrix P is the null component of A (PV = 0)
    P = project_matrix_onto_subspace(A, LN_V)
    # assert torch.allclose(P@V, torch.zeros_like(V), atol=1e-7, rtol=1e-7)
    return A - P

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

def check_grad_backprop(model):
    for name, parms in model.named_parameters():
        print(
            '-->name:', name, 
            '-->grad_requirs:', parms.requires_grad, 
            '--weight', torch.mean(parms.data), 
            ' -->grad_value:', torch.mean(parms.grad)
        )





###############################################################################
###############################################################################
###############################################################################

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)




