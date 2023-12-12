"""  v8. attention attribution across all heads (& layers?)

TODO not 100% finished. what's left to do: how to apply key padding mask in flashattention?
v7. pan-cancer
omitting zero valued genes for both pretraining & fine-tuning as scGPT & scBERT did
"""

from typing import Optional, List, Tuple, Union
import torch
from torch import nn
import numpy as np
from flash_attn import flash_attn_func
import utils

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            n_heads: int,
            d_e: int,
            d_q: Optional[int] = None,
            d_k: Optional[int] = None,
            d_v: Optional[int] = None,
            dropout: Optional[float] = 0.5,
        ) -> torch.Tensor:
        super().__init__()

        self.d_e = d_e
        d_q = d_e if d_q is None else d_q
        d_k = d_e if d_k is None else d_k
        d_v = d_e if d_v is None else d_v
        assert d_q == d_k

        self.mha = MultiHeadAttention(
            n_heads, 
            d_e,
            d_q,
            d_k,
            d_v,
            dropout,
        )

        self.fc = nn.Linear(d_e, d_e)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_e)

    def forward(
        self,
        x_emb: torch.Tensor,
        x_emb_cond: torch.Tensor,
        return_attn: Optional[bool] = False,
        A_h: Optional[torch.Tensor] = None,
        head_idx: Optional[int] = None,
    ):
        r"""
        Args:
            x_emb (torch.Tensor): The query data embedding.
            x_emb_cond (torch.Tensor): The condition data (keys and values)
                embedding. If the same as x_emb, then perform self-attention,
                otherwise cross-attention.
            A_h (torch.Tensor): A fixed attention matrix at a particular head
                at a particular layer for MHA IG attr. Default: None.
            head_idx (int): The index of the head to use for MHA IG attr.

        Shape:
            - x_emb: :math:`(batch_size, n_tokens, d_e)`
            - x_emb_cond: :math:`(batch_size, n_token_cond, d_e)`
            - A_h: :math:`(n_tokens, n_token_cond)`

        Note:
            - Dropout should happen before adding the residual connection 
                (ref: https://github.com/feather-ai/transformers-tutorial/blob/main/layers/residual_layer_norm.py)
            - 
        """

        residual = x_emb
        output, A = self.mha(
            x_emb=x_emb, 
            x_emb_cond=x_emb_cond, 
            A_h=A_h, 
            head_idx=head_idx,
            return_attn=return_attn,
        )
        output = self.layer_norm(output + residual)
        residual = output
        output = self.dropout(self.fc(output))
        output = self.layer_norm(residual + output)

        return output if not return_attn else (output, A)

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads: int,
            d_e: int,
            d_q: Optional[int] = None,
            d_k: Optional[int] = None,
            d_v: Optional[int] = None,
            dropout: Optional[float] = 0.5,
            tensor_dtype: torch.dtype = torch.float32,
        ):
        r"""
        Args:
            n_heads (int): The number of heads in the multi-head attention.
            d_e (int): The embedding dimension.
            d_q (int): The dimension of the query embedding.
            d_k (int): The dimension of the key embedding.
            d_v (int): The dimension of the value embedding.
            dropout (float): The dropout rate.

        """

        super().__init__()
        self.tensor_dtype = tensor_dtype

        self.n_heads = n_heads
        self.d_e = d_e
        self.d_q = d_e if d_q is None else d_q
        self.d_k = d_e if d_k is None else d_k
        self.d_v = d_e if d_v is None else d_v
        assert d_q == d_k
        self.W_Q = nn.Linear(d_e, self.d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_e, self.d_q * n_heads, bias=False)
        self.W_V = nn.Linear(d_e, self.d_v * n_heads, bias=False)

        self.fc = nn.Linear(n_heads * self.d_v, d_e, bias=False)
        self.dropout = nn.Dropout(dropout)


    def forward(
            self, 
            x_emb: torch.Tensor,
            x_emb_cond: torch.Tensor,
            return_attn: Optional[bool] = False,
            A_mask: Optional[torch.Tensor] = None,
            A_h: Optional[torch.Tensor] = None,
            head_idx: Optional[int] = None,
        ):
        """

        Args:
            x_emb (torch.Tensor): The query data embedding.
            x_emb_cond (torch.Tensor): The condition data (keys and values) 
                embedding. If the same as x_emb, then perform self-attention, 
                otherwise cross-attention.
            A_mask (torch.Tensor): The mask for the attention matrix. Default: 
                None. 
            A_h (torch.Tensor): User can fix the attention matrix at a 
                particular head for IG attr. Default: None.
            head_idx (int): index of the head to use for IG attr. Default: None.
        Shape:
            - x_emb: :math:`(N, L, E)`
            - x_emb_cond: :math:`(N, L_cond, E)`
            - A_h: :math:`(L, L_cond)`
            - A_mask: :math:`(N, L, L_cond)`
        """
        use_flash_attn = False if next(self.parameters()).device.type == 'cpu' else True

        batch_size = x_emb.shape[0]

        # linear projection
        self.Q = self.W_Q(x_emb) \
            .view(batch_size, -1, self.n_heads, self.d_q) \
            .transpose(1, 2) # (N, L, E) -> (N, L, H * E_q) -> (N, L, H, E_q) -> (N, H, L, E_q)
        self.K = self.W_K(x_emb_cond) \
            .view(batch_size, -1, self.n_heads, self.d_k) \
            .transpose(1, 2)
        self.V = self.W_V(x_emb_cond) \
            .view(batch_size, -1, self.n_heads, self.d_v) \
            .transpose(1, 2)

        # scaled dot-product attention
        ## TODO I doubt the correctness of the engineering trick of head concat ... Similar to the attnattr .view transformation of A into a vector, I may need to further confirm this
        if A_h is None:
            A = None
            if return_attn:
                with torch.no_grad(): 
                    # NOTE takes a lot of memory
                    scores = torch.matmul(self.Q, self.K.transpose(-1, -2)) / np.sqrt(self.d_k)
                    A = nn.Softmax(dim=-1)(scores) # A: (N, H, L, L_cond)
            ## Option: flashattn # NOTE MUST USE CUDA
            if use_flash_attn:
                self.Q = self.Q.permute(0, 2, 1, 3).to(torch.bfloat16) # (N, H, L, E) -> (N, L, H, E)
                self.K = self.K.permute(0, 2, 1, 3).to(torch.bfloat16)
                self.V = self.V.permute(0, 2, 1, 3).to(torch.bfloat16)
                assert self.d_k == self.d_v # ? TODO
                """ Notes about flash_attn_func:
                - emb_dim is 192 maximum per head for our GPU. Can reach 256 for certain other GPUs
                - n_heads < 32
                - it seems flash_attn_func is equivalent to: torch.softmax(q@k.transpose(2, 3) / torch.sqrt(torch.tensor(E)), dim=-1)@v. But to be 100% certain, I need to test in the backprop training setting
                - only support bfloat16 and float16 dtype
                - support L_q != L_k, L_q != L_v, L_k == L_v
                - requires d_v == d_k == d_q (TODO seems)
                """
                context = flash_attn_func(
                    q = self.Q,
                    k = self.K,
                    v = self.V,
                    dropout_p = 0, # TODO NOTE what's this??
                ) # (N, L, H, E) -> (N, L, H, E)
                context = context.to(self.tensor_dtype)
                context = context.permute(0, 2, 1, 3) # (N, L, H, E) -> (N, H, L, E)
            else:
                ### Option: normal
                scores = torch.matmul(self.Q, self.K.transpose(-1, -2)) / np.sqrt(self.d_k)
                A = nn.Softmax(dim=-1)(scores) # A: (N, H, L, n_token_cond)
                A[0, head_idx] = A_h # IG attr. Since can only manipulate one instance at a time, the batch_size must be 1.
                context = torch.matmul(A, self.V) # context: (N, H, L, E_v)
            
            context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # (N, H, L_q, E_v) -> (N, L_q, H, E_v) -> (N, L_q, H * E_v)
            output = self.dropout(self.fc(context)) # d_v to d_e: (N, L_q, H * E_v) -> (N, L, E)
        elif A_h is not None: # Analysis mode: Get Attn Attr
            assert head_idx is not None
            assert return_attn
            ## Option: normal
            scores = torch.matmul(self.Q, self.K.transpose(-1, -2)) / np.sqrt(self.d_k)
            A = nn.Softmax(dim=-1)(scores) # A: (N, H, L, n_token_cond)
            A[0, head_idx] = A_h # IG attr. Since can only manipulate one instance at a time, the batch_size must be 1.
            context = torch.matmul(A, self.V) # context: (N, H, L, E_v)
            
            context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # (N, H, L_v, E_v) -> (N, L_v, H, E_v) -> (N, L_v, H * E_v)
            output = self.dropout(self.fc(context)) # d_v to d_e: (N, L, H * E_v) -> (N, L, E)

        return output, A

class Module1(torch.nn.Module):
    r"""Module1: omics embedding + attention
    
    
    """

    def __init__(
        self, 
        n_tokens: int,
        d_e: int,
        n_heads: Optional[int] = 8,
        n_encoders: Optional[int] = 12,
        dropout: Optional[float] = 0.5,
        tensor_dtype: torch.dtype = torch.float32,
    ):
        r"""
        
        Args:
            n_tokens (int): Sequence length; The number of features.
            d_e (int): The embedding dimension.
            n_heads (int): The number of heads in the multi-head attention.
        """
        super().__init__()
        self.tensor_dtype = tensor_dtype 

        self.n_tokens = n_tokens
        self.d_e = d_e
        self.n_heads = n_heads

        self.token_emb_vocab = self.build_vocab(n_tokens, d_e)

        # TODO specify d_q, d_k, d_v to reduce computational cost?
        self.encoders = nn.Sequential(*[
            TransformerEncoder(
                n_heads = n_heads,
                d_e = d_e,
                dropout=dropout,
            ) for _ in range(n_encoders)
        ])


    def get_attention(
        self,
        x: torch.Tensor,
        x_cond: torch.Tensor,
    ):
        r""" Get the standard attention matrix from MHA.
        
        Args:
            x (torch.Tensor, numpy.ndarray)
        Shape:
            - Input: x :math:`(N, L)` where N is the batch size, L is the 
                sequence length.
            - Output: A :math:`(N, H, L, L_cond)` where N is the batch size, L is the 
                sequence length, and H is the number of heads.
        """
        x_emb = self._build_emb(x) # (N, L) -> (N, L, E)
        x_emb_cond = self._build_emb(x_cond) # (N, L) -> (N, L, E)
        
        # TODO accommodate for cross-attention option later
        # TODO Accommodate for other layers as well later (but which layer? need to think)
        self.eval()
        A = self.encoders[0].mha(
            x_emb = x_emb, 
            x_emb_cond = x_emb_cond,
        )[1].detach()
        return A

    def build_vocab(
        self,
        n_tokens: int,
        d_e: int,
        pretrained_token_emb_weight: torch.Tensor = None,
    ):
        r""" Construct embedding by integrating omics value into identity 
        embedding vocabulary.

        Args:
            x (torch.Tensor, numpy.ndarray):
        """

        if pretrained_token_emb_weight is not None:
            assert pretrained_token_emb_weight.shape == (n_tokens + 1, d_e)
            token_emb_vocab = torch.nn.Embedding.from_pretrained(
                embeddings = pretrained_token_emb_weight,
                freeze = True,
                padding_idx = 0
            )
        else:
            token_emb_vocab = torch.nn.Embedding(
                num_embeddings = n_tokens + 1,
                embedding_dim = d_e,
                padding_idx = 0
            )

        return token_emb_vocab
    
    def _build_emb(
            self,
            x: torch.Tensor,
        ):
        r"""Incorporate x (values) into token embedding, generating an input 
        embedding.

        Shape:
            - Input: x :math:`(N, L)` where N is the batch size, L is the 
                sequence length.
            - Output: x_emb :math:`(N, L, E)` where N is the batch size, L is the 
                sequence length, and E is the embedding dimension.
        
        """
        assert x.shape[1] == self.n_tokens
        # token embedding
        x_token_idx = torch.arange(x.shape[1]).repeat(x.shape[0], 1) + 1
        x_token_idx[x==0] = 0 # pad token for zero values
        x_token_idx = x_token_idx.to(self.token_emb_vocab.weight.device)
        x_token_emb = self.token_emb_vocab(x_token_idx) # x_token_emb: (N, L, E)

        # incorporate expression values to token embedding
        ## x_emb = x.unsqueeze(2) * x_token # (N, L) -> (N, L, 1); (N, L, 1) * (L, E) -> (N, L, E)
        x_emb = x.unsqueeze(2) * x_token_emb # (N, L, 1) * (N, L, E) -> (N, L, E)

        return x_emb

    def forward(
            self, 
            x: torch.Tensor,
        ):
        r"""
        Args:
            x (torch.Tensor, numpy.ndarray): 
        
        Shape:
            N is the batch size, L is the sequence length.
            - Input: :math:`(N, L, E)`
            - Output: :math:`(N, L, E)`
            
        """
        x_emb = self._build_emb(x) # x_emb: (N, L, E)

        output = x_emb
        for encoder in self.encoders:
            output = encoder(
                x_emb=output, 
                x_emb_cond=output,
            ) # TODO cross-attention?

        return output # shape same as x_emb

# TODO
class Module2(torch.nn.Module):
    r"""
    
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        NotImplemented

# TODO currently serving as Module3
class MLP(torch.nn.Module):
    r""" A customary MLP module.
    
    """
    def __init__(
        self, 
        channels: list,
        dropout: Optional[float] = 0.2,
    ):
        super().__init__()

        if len(channels) < 2:
            raise ValueError("The list of dimensions must contain at least two values.")
        
        self.layers = torch.nn.Sequential()
        for i in range(len(channels) - 1):
            self.layers.append(
                torch.nn.Linear(channels[i], channels[i + 1],)
            )
            if i < len(channels) - 2:
                self.layers.append(torch.nn.GELU()) # same as BERT
                self.layers.append(torch.nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MMBL(torch.nn.Module):
    def __init__(
        self,
        n_tokens: int,
        d_e: int,
        pretrain: bool,
        n_heads: Optional[int] = 1,
        n_classes: int = 1,
        pretrained_token_emb_weight: Optional[torch.Tensor] = None,
        n_encoders: Optional[int] = 1,
        dropout: Optional[float] = 0.5,
        tensor_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()


        self.tensor_dtype = tensor_dtype
        self.pretrain = pretrain

        self.n_tokens = n_tokens
        self.d_e = d_e

        self.module1 = Module1(
            n_tokens,
            d_e,
            n_heads=n_heads,
            n_encoders=n_encoders,
            dropout=dropout,
        )

        if pretrain:
            # for masked tokens
            self.token_predictor = MLP(
                channels = [d_e, 512, 1], # TODO
            )
        else:
            if pretrained_token_emb_weight is not None:
                self.module1.token_emb_vocab = self.module1.build_vocab(
                    n_tokens = n_tokens,
                    d_e = d_e,
                    pretrained_token_emb_weight = pretrained_token_emb_weight,
                )

            # TODO module2 
            self.last_linear_layer = torch.nn.Linear(d_e, 1)
            assert n_classes is not None
            self.module3 = MLP(channels=[n_tokens, n_tokens, n_classes])

        # TEMP NOTE  to(device) will move all sub nn.Modules and nn.Parameters to device, but will not move tensors created in those modules' forward to device


    def forward(
        self, 
        x: torch.Tensor,
    ):
        r"""
        Args:
            - x (torch.Tensor): The input data.
            pretrain_mode (bool): Whether to use the model in pretrain mode.
                Default: False.
        Shape:
            - x: :math:`(N, L)`
            - Output: :math:`(N, 1)`
        """
        if self.pretrain:
            x_context = self.module1(
                x,
            ) # x_context: (N, L, E)
            x_pred = self.token_predictor(x_context).squeeze(2) # (N, L, E) -> (N, L, 1) -> (N, L)
            return x_pred

        x_context = self.module1(
            x, 
        )

        # TODO convert x_embeddings to a vector for downstream tasks
        # Method 1: 
        # x_tran = self.att_output.sum(dim=2) # NOTE 2 (N, L) is much better than 1 (N, E) (latter results in no loss decrease) # NOTE 2: .sum is much better than .mean (why?) when not using MLP as last linear layer. NOTE 3: do not use (dim=2) together with layernorm
        # Method 1' [https://academic.oup.com/nar/article/49/13/e77/6266414]
        # x_tran = (self.att_output - self.att_output.mean(dim=1).unsqueeze(1)) / self.att_output.std(dim=1).unsqueeze(1)
        # Method 2: just a linear layer. # NOTE best. better than method 1. method 1 with add+norm results in non-decreasing loss
        x_transformed = self.last_linear_layer(x_context).squeeze(2) # (N, L, E) -> (N, L, 1) -> (N, L)

        logits = self.module3(x_transformed) # (N, L) -> (N, 1)

        return logits
