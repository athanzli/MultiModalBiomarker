from typing import Optional, List, Tuple, Union
import torch
from torch import nn
import numpy as np

import utils

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            n_heads: int,
            emb_dim: int,
            E_q: Optional[int] = None,
            E_k: Optional[int] = None,
            E_v: Optional[int] = None,
            drop_rate: Optional[float] = 0.2,
            device: Optional[Union[torch.device, str]] = torch.device('cpu')
        ) -> torch.Tensor:
        super().__init__()

        self.emb_dim = emb_dim
        E_q = emb_dim if E_q is None else E_q
        E_k = emb_dim if E_k is None else E_k
        E_v = emb_dim if E_v is None else E_v
        assert E_q == E_k
        self.device = device

        self.mha = MultiHeadAttention(
            n_heads, 
            emb_dim,
            E_q,
            E_k,
            E_v,
            drop_rate,
            device,
        )

        self.fc = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        query,
        key,
        value
    ):
        r"""

        Inputs:
            query: [batch_size, L_q, emb_dim]
            key: [batch_size, L_k, emb_dim]
            value: [batch_size, L_v(=L_k), emb_dim]
        Outputs:
            output: [batch_size, L_v, emb_dim]
        
        Note:
            - Dropout should happen before adding the residual connection 
                (ref: https://github.com/feather-ai/transformers-tutorial/blob/main/layers/residual_layer_norm.py)
            - 
        """
        residual = query
        output, A = self.mha(query, key, value)
        output = nn.LayerNorm(self.emb_dim).to(self.device)(output + residual) # TODO are you sure here should be (+ query) ?
        residual = output
        output = self.dropout(self.fc(output))
        output = nn.LayerNorm(self.emb_dim).to(self.device)(residual + output)

        return output

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads: int,
            emb_dim: int,
            E_q: Optional[int] = None,
            E_k: Optional[int] = None,
            E_v: Optional[int] = None,
            drop_rate: Optional[float] = 0.2,
            device: Optional[Union[torch.device, str]] = torch.device('cpu')
        ):
        r"""
        
        Example:
            >>> attn = MultiHeadAttention(n_heads=2, emb_dim=4)
            >>> x_emb = torch.randn(2, 3, 4)
            >>> attn(x_emb, x_emb, x_emb)
            >>> context, attn = attn(x_emb, x_emb, x_emb)
        """

        super().__init__()
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.E_q = emb_dim if E_q is None else E_q
        self.E_k = emb_dim if E_k is None else E_k
        self.E_v = emb_dim if E_v is None else E_v
        assert E_q == E_k
        self.W_Q = nn.Linear(emb_dim, self.E_k * n_heads, bias=False)
        self.W_K = nn.Linear(emb_dim, self.E_q * n_heads, bias=False)
        self.W_V = nn.Linear(emb_dim, self.E_v * n_heads, bias=False)

        self.fc = nn.Linear(n_heads * self.E_v, emb_dim, bias=False)
        self.dropout = nn.Dropout(drop_rate)

    def forward(
            self, 
            query, 
            key, 
            value, 
            att_mask: Optional[torch.tensor] = None,
        ):
        """
        Inputs:
            query: [batch_size, L_q, emb_dim]
            key: [batch_size, L_k, emb_dim]
            value: [batch_size, L_v(=L_k), emb_dim]
            att_mask: [batch_size, seq_len, seq_len]
        Outputs:
            output: [batch_size, L_v, emb_dim]
            A: [batch_size, n_heads, L_q, L_k]
        """
        batch_size = query.shape[0]

        self.Q = self.W_Q(query) \
            .view(batch_size, -1, self.n_heads, self.E_q).transpose(1, 2)  # Q: [batch_size, n_heads, L_q, E_k]
        self.K = self.W_K(key) \
            .view(batch_size, -1, self.n_heads, self.E_k).transpose(1, 2)  # K: [batch_size, n_heads, L_k, E_k]
        self.V = self.W_V(value) \
            .view(batch_size, -1, self.n_heads, self.E_v).transpose(1, 2)  # V: [batch_size, n_heads, L_v(=L_k), E_v]

        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # Component: scaed dot-product attention
        scores = torch.matmul(self.Q, self.K.transpose(-1, -2)) / np.sqrt(self.E_k)
        if att_mask is not None:
            scores.masked_fill_(att_mask, -1e9)
        A = nn.Softmax(dim=-1)(scores) # A: [batch_size, n_heads, L_q, L_k]
        context = torch.matmul(A, self.V) # context: [batch_size, n_heads, L_q, E_v]
        
        # (N, n_heads, L_v, E_v) --> (N, L_v, n_heads, E_v) --> (N, L_v, n_heads * E_v)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.E_v)

        output = self.dropout(self.fc(context)) # (N, L_v, n_heads * E_v) --> (N, L_v, emb_dim)

        return output, A

class Module1(torch.nn.Module):
    r"""Module1: omics embedding + attention
    
    
    """

    def __init__(
        self, 
        seq_len: int,
        emb_dim: int,
        n_heads: Optional[int] = 1,
        n_encoders: Optional[int] = 1,
    ):
        r"""
        
        Args:
            seq_len (int): Sequence length; The number of features.
            emb_dim (int): The embedding dimension.
            n_heads (int): The number of heads in the multi-head attention.
        """
        super().__init__()

        self.seq_len = seq_len
        self.emb_dim = emb_dim

        self.G_emb_vocab = self.build_vocab(seq_len, emb_dim)

        self.encoder = nn.Sequential(*[
            TransformerEncoder(
                n_heads = n_heads,
                emb_dim = emb_dim,
            ) for _ in range(n_encoders)
        ])

    def effective_attention(
            self,
            x: Union[torch.tensor, np.ndarray],
        ):
        r""" Get the effective attention matrix from standard attention.


        Args:
            x (torch.tensor, numpy.ndarray): N is the batch size, L is the 
                sequence length.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x_emb = self._build_emb(x) # (N, L) --> (N, L, E)
        
        self.eval()
        A = self.encoder[0].mha(
            query = x_emb, 
            key = x_emb, 
            value = x_emb
        )[1].detach().squeeze(1) # (N, n_heads, L, E) --> (N, L, E)

        V = self.encoder[0].mha.V.detach().squeeze(1) # (N, n_heads, L, E) --> (N, L, E)

        # TODO might need speeding up
        A_eff = torch.zeros_like(A)
        for i in range(A.shape[0]):
            A_eff[i] = utils.effective_attention(
                A = A[i], 
                V = V[i]
            )

        return A_eff

    def build_vocab(
        self,
        seq_len: int,
        emb_dim: int,
        pretrained_vocb_weights: torch.Tensor = None,
    ):
        r""" Construct embedding by integrating omics value into identity 
        embedding vocabulary.

        Args:
            x (torch.tensor, numpy.ndarray): N is the batch size, L is the 
                sequence length.
        """

        if pretrained_vocb_weights is not None:
            pretrained_vocb_weights = torch.FloatTensor(pretrained_vocb_weights)
            G_emb_vocab = torch.nn.Embedding.from_pretrained(
                embeddings = pretrained_vocb_weights,
                freeze = True,
            )
        else:
            G_emb_vocab = torch.nn.Embedding( # ok yet unstable?
                num_embeddings = seq_len,
                embedding_dim = emb_dim,
            )
            # TODO DEL
            G_emb_vocab = torch.nn.Embedding.from_pretrained(
                embeddings = G_emb_vocab.weight.detach(),
                freeze = True,
            )

        return G_emb_vocab

    def _build_emb(
        self,
        x: Union[torch.tensor, np.ndarray],
    ):
        r"""Incorporate information in x into embedding
        
        """
        ###### Embedding - NOTE huge affect on feature attention score; though all good for final accuracy
        
        x_emb = x.unsqueeze(2) * self.G_emb_vocab(torch.arange(self.seq_len))
        
        # self.G_emb_vocab = one_hot(torch.arange(seq_len)).to(torch.float32) # bad. every feature has almost the same attention. maybe because of too many zero values
        # gene_em = x.unsqueeze(2) * self.G_emb_vocab
        
        # self.G_emb_vocab = torch.randn(seq_len, emb_dim) # good yet unstable for attention score
        # gene_em = x.unsqueeze(2) * self.G_emb_vocab
        
        # self.G_emb_vocab = torch.ones(seq_len, emb_dim) # bad for attention score
        # gene_em = x.unsqueeze(2) * self.G_emb_vocab
        
        # self.G_emb_vocab = torch.arange(seq_len).unsqueeze(dim=1).repeat(1, emb_dim).to(torch.float32) # bad for attention score (attention tends to focus on the high valued embeddings/features)
        # x_emb = x.unsqueeze(2) * self.G_emb_vocab

        # try the binning technique?

        return x_emb

    def forward(
        self, 
        x: Union[torch.tensor, np.ndarray],
        output_att_scores: bool = False,
    ):
        r"""
        Args:
            x (torch.tensor, numpy.ndarray): N is the batch size, L is the 
                target sequence length, and E is the embedding dimension.
            output_att_scores (bool): Whether to output the attention scores.
        
        Shape:
            - Input: :math:`(N, L, E)`
            - Output: :math:`(N, L, E)`
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        x_emb = self._build_emb(x) # self.x_emb: (N, L, E)

        output = x_emb
        for layer in self.encoder:
            output = layer(output, output, output)

        if output_att_scores:
            return output, output
        return output # (N, L)

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
                self.layers.append(torch.nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MMBL(torch.nn.Module):
    def __init__(
        self,
        seq_len: int,
        emb_dim: int,
        mlp_channels: List[int],
        n_heads: Optional[int] = 1,
        pretrained_vocb_weights: Optional[torch.Tensor] = None,
        n_encoders: Optional[int] = 1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.emb_dim = emb_dim

        self.module1 = Module1(seq_len, emb_dim, n_heads=n_heads)

        if pretrained_vocb_weights is not None:
            self.module1.G_emb_vocab = self.module1.build_vocab(
                seq_len = seq_len,
                emb_dim = emb_dim,
                pretrained_vocb_weights = pretrained_vocb_weights,
            )

        # TODO module2 
        self.last_linear_layer = torch.nn.Linear(emb_dim, 1)

        self.module3 = MLP(mlp_channels)

    def effective_attention(
            self, 
            x: Union[torch.tensor, np.ndarray],
        ):
        r"""
        
        Shape:
            - Input: :math:`(N, L)`
            - Output: :math:`(N, L, L)`
        """
        return self.module1.effective_attention(x)

    def forward(self, x):
        x_context = self.module1(x)

        # TODO convert x_embeddings to a vector for downstream tasks
        # Method 1: 
        # x_tran = self.att_output.sum(dim=2) # NOTE 2 (N, L) is much better than 1 (N, E) (latter results in no loss decrease) # NOTE 2: .sum is much better than .mean (why?) when not using MLP as last linear layer. NOTE 3: do not use (dim=2) together with layernorm
        # Method 1' [https://academic.oup.com/nar/article/49/13/e77/6266414]
        # x_tran = (self.att_output - self.att_output.mean(dim=1).unsqueeze(1)) / self.att_output.std(dim=1).unsqueeze(1)
        # Method 2: just a linear layer. # NOTE best. better than method 1. method 1 with add+norm results in non-decreasing loss
        x_transformed = self.last_linear_layer(x_context).squeeze(2) # (N, L, E) -> (N, L, 1) --> (N, L)

        logits = self.module3(x_transformed)

        return logits
    











