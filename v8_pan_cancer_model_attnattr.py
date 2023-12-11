# attn attr for pan cancer model. across all cancer types.
import torch
import numpy as np
import captum
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
###############################################################################
###############################################################################
# TODO what if multiple layers?
class MMBL_AttnAttr(torch.nn.Module):
    r"""An auxiliary model that, with a fixed input X, takes A as input and 
    ouptuts the pred results.

    """
    def __init__(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        x_cond: torch.Tensor,
        layer_idx: int,
        head_idx: int,
        n_tokens: int,
        n_token_cond: int,
        n_step: int,
        tensor_dtype: torch.dtype = torch.float32,
    ):
        r"""
        Args:
            - model (torch.nn.Module): The trained MMBL model.
            - x (torch.Tensor): The input data. Fixed upon creating instance.
            - layer_idx (int): The index of the layer to use. Must specify since
                the IG attr API of Captum can only manipuate one input tensor 
                at a time, which in our case is an attention matrix at a particular
                head at a particular layer.
            - head_idx (int): The index of the head to use. Must specify since
                the IG attr API of Captum can only manipuate one input tensor 
                at a time, which in our case is an attention matrix at a particular
                head at a particular layer
            - n_step (int): The number of steps to use for IG attr.
        """
        super().__init__()
        self.tensor_dtype = tensor_dtype

        self.model = model
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.n_tokens = n_tokens
        self.n_token_cond = n_token_cond
        self.n_step = n_step

        self.x_emb = self.model.module1._build_emb(x) # x_emb: (batch_size, n_tokens, d_e)
        self.x_emb_cond = self.model.module1._build_emb(x_cond) # x_emb_cond: (batch_size, n_tokens, d_e)

    def forward(
        self, 
        A: torch.Tensor,
    ):
        r"""
        
        Args:
            A (torch.Tensor): The attention matrix at a paticular head at a 
                particular layer.
        
        Shape:
            :math:`(1, L * L_cond)`. Captum IG attr automatically
                assumes rows of input are instances, and creates n_step * 1
                instances.
        """

        A = A.view(self.n_step, self.n_tokens, self.n_token_cond)

        # TODO only 1 layer currently
        encoder = self.model.module1.encoders[self.layer_idx]
        
        res = torch.zeros(self.n_step, 1) # TODO change according to task
        for i in range(self.n_step):
            x_context = encoder(
                x_emb = self.x_emb,
                x_emb_cond = self.x_emb_cond, # TODO!!!
                A_h = A[i],
                head_idx = self.head_idx,
                return_attn = False,
            )
            x_transformed = self.model.last_linear_layer(x_context).squeeze(2) # (N, L, E) -> (N, L, 1) -> (N, L)
            logits = self.model.module3(x_transformed) # (N, L) -> (N, (?)). according to task

            res[i] = logits

        return res

###############################################################################
###############################################################################
###############################################################################
"""
NOTE ig.attribute(inputs=inputs, n_step=n_step) 
    assumes inputs' rows are instances and columns are features. 
    It will first divide inputs into n_step * n_rows instances, then...
"""
def attn_attr(
        model, 
        X_input, 
        X_cond_input,
        n_step, 
        layer_idx=0, 
        multiply_by_A=True
    ):
    r"""
    Args:
        model: MMBL model.
        X_input: 
        n_step: integration steps for IG.
        layer_idx: 
        multiply_by_A: If True, multiply the attention matrix A by the 
            attribution matrix, else only return the attribution matrix.
            NOTE Seems results are similar... ; seems False would be better?? idk.
    """
    X_input = X_input.detach()
    X_cond_input = X_cond_input.detach()
    A = model.module1.get_attention(X_input, X_cond_input).detach() # (N, H, L, L_cond)
    A_attr = torch.zeros_like(A)
    for i in range(X_input.shape[0]):
        x = X_input[i]
        x_cond = X_cond_input[i]
        x = x.view(1, -1)
        x_cond = x_cond.view(1, -1)
        for j in range(model.module1.n_heads):
            model_attnattr = MMBL_AttnAttr(
                model = model,
                x = x,
                x_cond = x_cond,
                n_tokens = A[i, j].shape[0],
                n_tokens_cond = A[i, j].shape[1],
                n_step = n_step,
                layer_idx=0,
                head_idx=j, # NOTE
                tensor_dtype=model.tensor_dtype
            )
            ig = captum.attr.IntegratedGradients(
                model_attnattr.forward, 
                multiply_by_inputs=multiply_by_A # NOTE!
            ) # can only manipulate one A at a one particular layer l on one particular head h
            attr = ig.attribute(inputs=A[i, j].view(1, -1), n_steps=n_step) # (L, L_cond) -> (n_step, L * L_cond) - ig.attribute -> ?
            attr = attr.view(A[i, j].shape) # ? -> (L, L_cond)
            A_attr[i, j] = attr
    return A_attr

def attention_avg(A, sample_mask):
    r""" Get the average or median attention scores matrix for a particular category.
    
    """
    A_avg = np.median(A[sample_mask], axis=0)
    return A_avg

def two_sets_stats(set1, set2):
    intersection = set(set1) & set(set2)
    # union = set(set1) | set(set2)
    only_set1 = set(set1) - intersection
    only_set2 = set(set2) - intersection
    print("Intersection:", len(intersection), '. ', intersection)
    print("Only in set1:", len(only_set1), '. ', only_set1)
    print("Only in set2:", len(only_set2), '. ', only_set2)
    return intersection, only_set1, only_set2

def remove_dashdash(df):
    # TODO just a temp func
    idx=[]
    for i in range(df.shape[1]):
        if (df.iloc[:, i].values=="'--").sum() == df.shape[0]:
            idx.append(i)
    df = df.drop(columns=df.columns[idx])
    return df

from typing import Optional, List, Union
def feature_score_barplot(
        A, 
        mode,
        x_id: Optional[Union[np.ndarray, pd.Series]]=None,
        y_id: Optional[Union[np.ndarray, pd.Series]]=None,
        n_top=50,
        ):
    r""" Plot feature scores from an attention matrix. 
    TODO NOTE currently specific for concatenated miRNA-mRNA 

    Args:
        - A (pd.DataFrame, np.ndarray): Attention matrix of one sample or average of samples.
        - mode:
        - n_top (int): Only plot the top n_top features.
        - x_id (np.ndarray, pd.Series): Specify features of interest along axis 0.
        - y_id (np.ndarray, pd.Series): Specify features of interest along axis 1.
    Shape:
        - A: (n_tokens, n_tokens_cond)
    """

    A = A[np.ix_(np.array(x_id), np.array(y_id))]

    A = pd.DataFrame(
        data=A, 
        index=x_id.index, 
        columns=y_id.index
    )

    plt.figure(figsize=(10, 3))

    if mode == 'inter':
        A_stack = A.stack().reset_index()
        A_stack = A_stack.sort_values(by=0, ascending=False)
        A_stack['pair'] = A_stack['level_0'] + '_' + A_stack['level_1']
        plt.bar(range(n_top), A_stack[0][:n_top].values, align='center')
        plt.xticks(range(n_top), A_stack['pair'][:n_top], rotation='vertical')
    elif mode == 'intra':
        A_sorted = A.mean(axis=0).sort_values(ascending=False)
        plt.bar(range(n_top), A_sorted.values[:n_top], align='center')
        plt.xticks(range(n_top), A_sorted.index[:n_top], rotation='vertical')
    else:
        raise ValueError(f'{mode} for parameter mode is not supported.')

    plt.xlabel('Features')
    plt.ylabel('Scores')
    plt.show()

def att_cat(A, mask, lam=5):
    A_avg = np.median(A[mask], axis=0)

    # V0
    # feature_attn = np.mean(A_avg, axis=0)

    # # V1
    # ind = np.argsort(A_avg, axis=0)[-n_top:]
    # A_avg = np.take_along_axis(A_avg, ind, axis=0)
    # feature_attn = np.mean(A_avg, axis=0)

    # V2 NOTE seems improves feature identification performance?? TODO needs further confirmation through experiment
    feature_attn = np.mean(A_avg, axis=0) + lam*np.diag(A_avg)

    return A_avg, feature_attn

def plot_ft_imp(ft_names, ft_imp, title=None):
    sorted_idx = sorted(range(len(ft_imp)), key=lambda k: ft_imp[k], reverse=True)
    ft_names = [ft_names[i] for i in sorted_idx]
    ft_imp = [ft_imp[i] for i in sorted_idx]

    plt.figure(figsize=(10, 3))
    bars = plt.bar(range(len(ft_imp)), ft_imp, align='center')
    plt.xticks(range(len(ft_imp)), ft_names, rotation='vertical')

    plt.xlabel('Features')
    plt.ylabel('Scores')
    plt.title(title)
    plt.show()



###############################################################################
###############################################################################
###############################################################################

model = 

# ### a simple run
# model.eval()
# logits = model(X_tst)
# y_pred = logits.sigmoid().round().flatten()
# # mask = (y_tst==1) & (y_pred==1) # NOTE!! THIS IS NECESSARY (but not mainly due to & y_pred==1; because of specifying one class instead of using mixed classes!!!). without it, the result does not make sense.
# mask = (y_tst==0) & (y_pred==0) # NOTE!! THIS IS NECESSARY (but not mainly due to & y_pred==1; because of specifying one class instead of using mixed classes!!!). without it, the result does not make sense.
# X_input = X_tst[mask]
# ig = captum.attr.IntegratedGradients(model.forward, multiply_by_inputs=False)
# attr = ig.attribute(X_input, n_steps=50)
# ft_imp = attr.mean(dim=0)
# plot_ft_imp(ft_names[:10], ft_imp[:10])

### using AttrAtta
model.eval()
logits = model(X_tst)
y_pred = logits.sigmoid().round().flatten().detach().cpu()

# def multi_A_filter(
#     As: List[np.ndarray],
#     thres: float
# ):
#     for mat in As:

mask = (y_tst.cpu()==1) & (y_pred==1) # NOTE!! THIS IS NECESSARY (but not mainly due to & y_pred==1; because of specifying one class instead of using mixed classes!!!). without it, the result does not make sense.
X_input = X_tst.cpu()[mask]
A_attr = attn_attr(model, X_input=X_input, X_cond_input=X_input, n_step=50, layer_idx=0).abs() # NOTE!!!
A_attr = A_attr.mean(dim=1).mean(dim=0).cpu().numpy()
# ##### option 1 - look at single modalities
x_id = pd.Series(
    index=mrna.columns.values,
    data=np.arange(mrna.shape[1])
)
y_id = pd.Series(
    index=mrna.columns.values,
    data=np.arange(mrna.shape[1])
)
feature_score_barplot(A=A_attr, mode='intra', n_top=20, x_id=x_id, y_id=y_id)
# ##### option 2 - look at two modalities
# ## NOTE:  will be more reasonable to use upper right submatrix (mRNA attend to miRNA)
# x_id = pd.Series(
#     index=mrna.columns.values,
#     data=np.arange(0, mrna.shape[1])
# )
# y_id = pd.Series(
#     index=mirna.columns.values,
#     data=np.arange(mrna.shape[1], mrna.shape[1] + mirna.shape[1])
# )
# feature_score_barplot(A=A_attr, mode='inter', n_top=20, x_id=x_id, y_id=y_id)

# mask = (y_tst.cpu()==0) & (y_pred==0) # NOTE!! THIS IS NECESSARY (but not mainly due to & y_pred==1; because of specifying one class instead of using mixed classes!!!). without it, the result does not make sense.
# X_input = X_tst.cpu()[mask]
# A_attr = att_attr(model, X_input, n_step=50, layer_idx=0).abs() # NOTE!!!
# A_attr = A_attr.mean(dim=1).mean(dim=0).cpu().numpy()
# # ##### option 1 - look at single modalities
# x_id = pd.Series(
#     index=mrna.columns.values,
#     data=np.arange(mrna.shape[1])
# )
# y_id = pd.Series(
#     index=mrna.columns.values,
#     data=np.arange(mrna.shape[1])
# )
# feature_score_barplot(A=A_attr, mode='intra', n_top=20, x_id=x_id, y_id=y_id)
# ##### option 2 - look at two modalities
# ## NOTE:  will be more reasonable to use upper right submatrix (mRNA attend to miRNA)
# x_id = pd.Series(
#     index=mrna.columns.values,
#     data=np.arange(0, mrna.shape[1])
# )
# y_id = pd.Series(
#     index=mirna.columns.values,
#     data=np.arange(mrna.shape[1], mrna.shape[1] + mirna.shape[1])
# )
# feature_score_barplot(A=A_attr, mode='inter', n_top=20, x_id=x_id, y_id=y_id)
