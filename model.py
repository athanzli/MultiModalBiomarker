from typing import Optional, List, Tuple, Union
import torch
import numpy as np

class Module1(torch.nn.Module):
    r"""Module1: omics embedding + attention
    
    
    """

    def __init__(
        self, 
        n_genes: int,
        emb_dim: int,
        n_heads: Optional[int] = 1,
    ):
        super().__init__()

        self.n_genes = n_genes
        self.emb_dim = emb_dim

        self.gene_emb_vocab = torch.nn.Embedding(
            num_embeddings = n_genes,
            embedding_dim = emb_dim
        )
        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim = emb_dim,
            num_heads = n_heads,
            batch_first = True # NOTE
        )

    # NOTE debug
    # class Emb2VecNet(torch.nn.Module):
    #     def __init__(self, emb_dim):
    #         super().__init__()
    #         self.fc1 = torch.nn.Linear(emb_dim, int(emb_dim / 2))
    #         self.fc2 = torch.nn.Linear(int(emb_dim / 2), 1)
    #         self.act = torch.nn.ReLU()

    #     def forward(self, x):
    #         N, L, E = x.size()
    #         x = x.view(-1, E)
    #         x = self.act(self.fc1(x))
    #         x = self.fc2(x)
    #         x = x.view(N, L, -1) # (N, L, 1)
    #         x = x.squeeze(2) # (N, L)
    #         return x

    def forward(
        self, 
        x: Union[torch.tensor, np.ndarray],
        output_attn_scores: bool = False,
    ):
        r"""
        Args:
            x (torch.tensor, numpy.ndarray): N is the batch size, L is the 
                target sequence length, and E is the embedding dimension.
            output_attn_scores (bool): Whether to output the attention scores.
        
        Shape:
            - Input: :math:`(N, L, E)`
            - Output: :math:`(N, L, E)`
        """
        
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        # TODO incorporate gene expression level information into the embedding
        gene_emb = x.unsqueeze(2) * self.gene_emb_vocab(torch.arange(self.n_genes))
        
        attn_output, self.attn_scores = self.self_attn(
            query=gene_emb, 
            key=gene_emb, 
            value=gene_emb
        )

        # TODO convert gene embeddings to a vector representation of a sample for downstream tasks
        # Method 1: seems bad
        # x_tran = attn_output.mean(dim=2)
        # Method 2: MLP. 
        # self.net = self.Emb2VecNet(emb_dim = self.emb_dim)
        # x_tran = self.net(attn_output) # (N, L, E) -> (N, L, 1) --> (N, L)
        # or method 2' Conv (essentially same as MLP)
        self.conv = torch.nn.Conv1d(
            in_channels = self.emb_dim,
            out_channels = 1,
            kernel_size = 1,
        )
        x_tran = self.conv(attn_output.permute(0, 2, 1)).squeeze(1) # (N, L, E) -> (N, E, L) -> (N, 1, L) -> (N, L) 

        if output_attn_scores:
            return x_tran, self.attn_scores
        return x_tran

# currently serving as Module3
class MLP(torch.nn.Module):
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
        n_genes: int,
        emb_dim: int,
        mlp_channels: List[int],
        n_heads: Optional[int] = 1,
    ):
        super().__init__()

        self.n_genes = n_genes
        self.emb_dim = emb_dim

        self.module1 = Module1(n_genes, emb_dim, n_heads=n_heads)
        self.module3 = MLP(mlp_channels)

    def forward(self, x):
        x_tran = self.module1(x)
        logits = self.module3(x_tran)
        return logits
    
