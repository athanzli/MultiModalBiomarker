###############################################################################
# setup
###############################################################################
# %load_ext autoreload
# %autoreload 2

import os
import argparse
import logging

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import pandas as pd
import random, os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import balanced_accuracy_score
import lifelines
from sksurv.metrics import concordance_index_censored
import copy
import captum
#
from utils import *
from model_v8 import MMBL, MLP

DATA_PATH = '/home/che82/athan/MMBiomarker_data'
SAVE_PATH = './saved'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# # TODO need to modify for multi modalities
# def mask_non_zero_values(X, prop=0.1): # TODO 
#     r""" 
#     Returns:
#         - maksed_X: those chosen will be set to the median value of 
#             the row. TODO better options?
#         - mask_X: those chosen will be set to True.
#     """
#     X_masked = X.clone()
#     X_mask = torch.zeros_like(X, dtype=torch.bool)

#     for i, row in enumerate(X):
#         non_zero_indices = (row != 0).nonzero(as_tuple=True)[0]
#         num_to_mask = int(len(non_zero_indices) * prop)
#         indices_to_mask = non_zero_indices[torch.randperm(len(non_zero_indices))[:num_to_mask]]

#         median_value = torch.median(row[non_zero_indices]) # TODO
#         X_masked[i, indices_to_mask] = median_value
#         X_mask[i, indices_to_mask] = True

#     return X_masked, X_mask

# TODO TEMP A temp version that proportionally mask mRNA and miRNA separately
N_MRNA = 4222
N_MIRNA = 1773
def mask_non_zero_values(X, prop=0.1): # TODO 
    r""" 
    Returns:
        - maksed_X: those chosen will be set to the median value of 
            the row. TODO better options?
        - mask_X: those chosen will be set to True.
    """
    X_masked = X.clone()
    X_mask = torch.zeros_like(X, dtype=torch.bool)

    for i, row in enumerate(X):
        non_zero_indices = (row != 0).nonzero(as_tuple=True)[0]
        non_zero_indices_mrna = non_zero_indices[non_zero_indices < N_MRNA]
        non_zero_indices_mirna = non_zero_indices[non_zero_indices >= N_MRNA]

        num_to_mask = int(len(non_zero_indices) * prop)
        num_to_mask_mrna = int(num_to_mask * N_MRNA / (N_MRNA + N_MIRNA))
        num_to_mask_mirna = num_to_mask - num_to_mask_mrna

        indices_to_mask_mrna = non_zero_indices_mrna[torch.randperm(len(non_zero_indices_mrna))[:num_to_mask_mrna]]
        indices_to_mask_mirna = non_zero_indices_mirna[torch.randperm(len(non_zero_indices_mirna))[:num_to_mask_mirna]]

        median_value_mrna = torch.median(row[non_zero_indices_mrna]) # TODO
        median_value_mirna = torch.median(row[non_zero_indices_mirna]) # TODO
        X_masked[i, indices_to_mask_mrna] = median_value_mrna
        X_masked[i, indices_to_mask_mirna] = median_value_mirna

        X_mask[i, indices_to_mask_mrna] = True
        X_mask[i, indices_to_mask_mirna] = True

    return X_masked, X_mask

def load_data(task: str, modality: str, tcga_subset: str = None):
    if task == 'pretrain':
        X1 = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-preprocessed/mRNA_deg_0.05_1.csv", index_col=0)
        X2 = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-preprocessed/miRNA.csv", index_col=0)
        samples = np.intersect1d(X1.index, X2.index)
        X1 = X1.loc[samples]
        X2 = X2.loc[samples]
        y = None

    elif task in ['TCGA-project', 'metastasis', 'stage-classification']:
        # classification task
        # TODO !!
        X1 = pd.read_csv(DATA_PATH + f"/TCGA-PanCancerData-preprocessed/{task}/mRNA.csv", index_col=0)
        X2 = pd.read_csv(DATA_PATH + f"/TCGA-PanCancerData-preprocessed/{task}/miRNA.csv", index_col=0)
        y = np.load(DATA_PATH + f"/TCGA-PanCancerData-preprocessed/{task}/y.npy")
        # X1 = pd.read_csv(DATA_PATH + f"/TCGA-PanCancerData-preprocessed/{task}/mRNA_small.csv", index_col=0)
        # X2 = pd.read_csv(DATA_PATH + f"/TCGA-PanCancerData-preprocessed/{task}/miRNA_small.csv", index_col=0)
        # y = np.load(DATA_PATH + f"/TCGA-PanCancerData-preprocessed/{task}/y_small.npy")
        assert len(X2) == len(y)
        y = pd.DataFrame(y, index=X2.index)

    elif task == 'survival':
        X1 = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-preprocessed/survival/mRNA.csv", index_col=0)
        X2 = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-preprocessed/survival/miRNA.csv", index_col=0)
        # times = np.load(DATA_PATH + f"/TCGA-PanCancerData-preprocessed/{task}/times.npy")
        # events = np.load(DATA_PATH + f"/TCGA-PanCancerData-preprocessed/{task}/events.npy")
        sv = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-preprocessed/survival/survival.csv", index_col=0)
        assert (X1.index == sv.index).all()
        y = (sv['T'], sv['E'])

    assert (X1.index == X2.index).all()
    X12 = pd.concat([X1, X2], axis=1)

    if modality == 'mRNA+miRNA':
        X = X12
    elif modality == 'mRNA':
        X = X1
    elif modality == 'miRNA':
        X = X2

    # # # TODO DEL JUST TEMP FOR
    # y_tmp = y[0] if isinstance(y, Tuple) else y
    # if dist.get_rank() == 0:
    #     print(f"Before taking TCGA subset:")
    #     print(f"X and y shape: {X.shape}, {y_tmp.shape}")
    #     if task != 'survival':
    #         print(f"y classes: {np.unique(y_tmp.values, return_counts=True)}")

    # TODO NOTE just for TCGA. change accordingly
    if tcga_subset is not None:
        assert y is not None, "Pretraining task does not support choosing TCGA subset."
        if len(X.index[0].split('-')) == 4:
            tcga_proj_labels = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-preprocessed/TCGA-project-label-for-sample.csv", index_col=0)
        elif len(X.index[0].split('-')) == 3:
            tcga_proj_labels = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-preprocessed/TCGA-project-label-for-patient.csv", index_col=0)
        else:
            raise ValueError("Unknown index format.")
        sel = tcga_proj_labels[tcga_proj_labels.values == tcga_subset].index
        sel = np.intersect1d(X.index, sel)
        X = X.loc[sel]
        y = y.loc[sel] if (not isinstance(y, Tuple)) else (y[0].loc[sel], y[1].loc[sel])

        # # TODO DEL JUST TEMP FOR 
        y_tmp = y[0] if isinstance(y, Tuple) else y
        if dist.get_rank() == 0:
            print(f"After taking TCGA subset:")
            print(f"X and y shape: {X.shape}, {y_tmp.shape}")
            if task != 'survival':
                print(f"y classes: {np.unique(y_tmp.values, return_counts=True)}")
            assert np.unique(y_tmp.values).shape[0] > 1, "Only one class in the subset."

    return X, y

class CustomTensorDataset(Dataset):
    r"""
    For both pretrain and fine-tuning.
    """
    def __init__(self, X, y: torch.Tensor = None):
        assert X is not None
        if y is not None:
            if isinstance(X, Tuple) & isinstance(y, Tuple):
                assert all(len(x) == len(y) for (x, y) in (X, y))
            elif isinstance(X, Tuple) & (not isinstance(y, Tuple)):
                assert all(len(x) == len(y) for x in X)
            elif (not isinstance(X, Tuple)) & (isinstance(y, Tuple)):
                assert all(len(X) == len(y) for y in y)
            else:
                assert len(X) == len(y)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X if isinstance(self.X, torch.Tensor) else self.X[0])

    def __getitem__(self, idx):
        x = tuple(x[idx] for x in self.X) if isinstance(self.X, Tuple) else self.X[idx]
        if self.y is None:
            y = None
        elif isinstance(self.y, Tuple):
            y = tuple(y[idx] for y in self.y)
        else:
            y = self.y[idx]
        return (x, y) if y is not None else x

def prep_dataloader(
    X: Union[torch.Tensor, Tuple[torch.Tensor]],
    batch_size: int,
    y: Optional[Union[torch.Tensor, Tuple[torch.Tensor]]] = None,
    sampler: bool = False
):
    dataset = CustomTensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=DistributedSampler(dataset) if sampler else None,
        shuffle=False if sampler else True,
        pin_memory=False, # if True, error?
    )
    return data_loader

def prep_data(
    task: str,
    batch_size: int,
    tcga_subset: str,
    modality: str = 'mRNA+miRNA',
    tensor_dtype: torch.dtype = torch.float32,
    val_data: bool = True,
    use_sampler: bool = False,
):
    if dist.get_rank() == 0:
        logging.info("Reading data...")
    X, y = load_data(task, modality=modality, tcga_subset=tcga_subset)
    if task == 'survival':
        times, events = y
    
    if dist.get_rank() == 0:
        logging.info(f"Reading data complete. Data shape: {X.shape}")

    # save token info
    token_idx = pd.DataFrame(
        index = np.insert(arr=X.columns.values.astype(str), obj=0, values='<pad>'),
        data = np.arange(X.shape[1] + 1),
        columns = ['index'] # index in model's torch.nn.Embedding
    )
    token_idx.to_csv(f'./saved/token_idx_{modality}.csv')
    
    # data loader
    X = torch.tensor(
        X.values if isinstance(X, pd.DataFrame) else X, 
        dtype=tensor_dtype)
    val_loader = None

    if task == 'pretrain':
        val_size = 0.03 # TODO
        X_trn, X_val = train_test_split(X, test_size=val_size, random_state=42)
        X_masked, X_mask = mask_non_zero_values(X_trn) # TODO
        trn_loader = prep_dataloader(
            X = (X_masked, X_trn, X_mask), # TODO
            batch_size = batch_size,
            sampler=True if use_sampler else False
        )
        if val_data:
            X_masked, X_mask = mask_non_zero_values(X_val) # TODO
            val_loader = prep_dataloader(
                X = (X_masked, X_val, X_mask), # TODO
                batch_size = batch_size,
            )

    elif task in ['TCGA-project', 'metastasis', 'stage-classification']:
        """ Classification.
        
        """
        print("POS 0000") # TODO DEL

        val_size = 0.1 if task == 'stage-classification' else 0.1 # TODO
        y = torch.tensor(y.values.flatten(), dtype=torch.int64)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
        train_idx, val_idx = next(sss.split(X, y))
        X_trn = X[train_idx]
        X_val = X[val_idx]
        y_trn = y[train_idx]
        y_val = y[val_idx]

        # TODO DEL
        # if dist.get_rank() == 0:
        print(f"Class distributions, trn: {np.unique(y_trn, return_counts=True)}, val: {np.unique(y_val, return_counts=True)}")


        trn_loader = prep_dataloader(
            X = X_trn,
            y = y_trn,
            batch_size = batch_size,
            sampler=True if use_sampler else False
        )
        if val_data:
            val_loader = prep_dataloader(
                X = X_val,
                y = y_val,
                batch_size = batch_size,
            )


        
    elif task == 'survival':
        r"""
        Divide into discrete time intervals.

        T = {t_0, t_1, ..., t_n}
        t_0 = 0
        t_n = max(T)
        [t_i, t_{i+1}) is the i-th interval
        """
        sv = pd.DataFrame({
            'T':times,
            'E':events,
        })
        y = np.zeros(sv.shape[0], dtype=np.int64)
        # breaks
        qt = np.quantile(sv['T'].values, [0.25, 0.5, 0.75])
        brks = np.insert(arr=qt, obj=[0, 3], values=[np.min(sv['T'].values), np.max(sv['T'].values)])
        mid_points = (brks[1:] + brks[:-1]) / 2
        mid_points = np.insert(arr=mid_points, obj=[0, 4], values=[np.min(sv['T'].values), np.max(sv['T'].values)])
        for i in range(len(brks) - 1):
            y[((sv['E'] == 1) & (sv['T'] >= brks[i]) & (sv['T'] < brks[i+1])).values] = i
        for i in range(len(mid_points) - 1):
            y[((sv['E'] == 0) & (sv['T'] >= mid_points[i]) & (sv['T'] < mid_points[i+1])).values] = i - 1 # if survives past the mid point of the interval, then it is considered to have survived the interval; -1 is assigned to patients who have not survived past the first interval, so that in the survival loss function, ...
        y[np.where(sv['T']==sv['T'].max())[0][0]] = len(brks) - 2 # the max time point is not considered in the for loop
        
        val_size = 0.1 # TODO. important for TCGA-XXXX survival task. some project will be out of range if this value is too small.
        
        y = torch.tensor(y, dtype=torch.int64)
        e = torch.tensor(sv['E'].values, dtype=torch.int64)
        t = torch.tensor(sv['T'].values, dtype=tensor_dtype)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42) # TODO SEED?
        train_idx, val_idx = next(sss.split(X, y))
        X_trn = X[train_idx]
        X_val = X[val_idx]
        y_trn = y[train_idx]
        y_val = y[val_idx]
        e_trn = e[train_idx]
        e_val = e[val_idx]
        t_trn = t[train_idx]
        t_val = t[val_idx]

        # TODO DEL ENSURE THE SAME ACROSS ALL RUNS.
        if dist.get_rank() == 0:
            print(f"val idx: {val_idx}")
        
        trn_loader = prep_dataloader(
            X = X_trn,
            y = (y_trn, e_trn, t_trn),
            batch_size = batch_size,
            sampler=True if use_sampler else False
        )
        if val_data:
            val_loader = prep_dataloader(
                X = X_val,
                y = (y_val, e_val, t_val),
                batch_size = batch_size,
            )

    # TODO DEL
    # if dist.get_rank() == 0:
    logging.info(f"Trn: {len(trn_loader.dataset)}, Val: {len(val_loader.dataset) if val_loader is not None else 0}")

    return trn_loader, val_loader

### hyperparameters TODO
BATCH_SIZE = 32
MAX_EPOCH = 10
EMB_DIM = 192 # flashattn max for non{...} GPU
LAYERS = 1
HEADS = 4
DROP_RATE = 0.2 # NOTE

###############################################################################
# training
###############################################################################

class Trainer:
    def __init__(
        self,
        task: str,
        modality: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        trn_loader: DataLoader,
        val_loader: DataLoader,
        loss_func: torch.nn.modules.loss._Loss,
        gpu_id: int,
        default_ddp_gpu_id : int = 0,
        tcga_subset: str = None
    ) -> None:
        self.task = task
        self.modality = modality
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gpu_id = gpu_id
        self.default_ddp_gpu_id = default_ddp_gpu_id 
        self.trn_loader = trn_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_func
        self.batch_size = self.trn_loader.batch_size

        self.task_metric = {
            "pretrain": "MSE loss",
            "TCGA-project": "balanced accuracy", 
            "metastasis": "balanced accuracy",
            "stage-classification": "balanced accuracy",
            "survival-binary": "balanced accuracy",
            "survival": "c-index"}[self.task]
        self.metric_op = {
            "pretrain": "min",
            "TCGA-project": "max", 
            "metastasis": "max",
            "stage-classification": "max",
            "survival-binary": "max",
            "survival": "max"}[self.task]

        if next(self.model.module.module1.token_emb_vocab.parameters()).requires_grad == False:
            self.with_pretrained = 'with_pretrained'
        else:
            self.with_pretrained = ''

        self.tcga_subset = tcga_subset if tcga_subset is not None else 'all'

        if self.task != 'pretrain':
            y_trn = self.trn_loader.dataset.y[0] if isinstance(self.trn_loader.dataset.y, Tuple) else self.trn_loader.dataset.y
            if self.val_loader.dataset.y is not None:
                y_val = self.val_loader.dataset.y[0] if isinstance(self.val_loader.dataset.y, Tuple) else self.val_loader.dataset.y
                y = torch.cat([y_trn, y_val], dim=0)
            self.n_classes = torch.unique(y).shape[0]
            if (self.task == 'survival') & (-1 in y):
                self.n_classes -= 1

    def aggr_across_gpus(self, value, op):
        r""" all_reduce serves as a blocking point across all processes.

        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value).cuda(self.gpu_id)
        if op == 'avg':
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            return value.item() / dist.get_world_size()
        elif op == 'max':
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            return value.item()
        elif op == 'min': # TODO disc.ReduceOp.MIN CAUSES THE NONE PROBLEM!
            dist.all_reduce(value, op=dist.ReduceOp.MIN)
            return value.item()
        
    def _pretrain_run_epoch(
        self, 
    ):
        total_loss = 0
        for i, x in enumerate(self.trn_loader):
            x_masked, x_true, x_mask = x[0].cuda(self.gpu_id), x[1].cuda(self.gpu_id), x[2].cuda(self.gpu_id)
            self.optimizer.zero_grad()
            x_pred = self.model(x_masked)
            loss = self.loss_fn(x_pred[x_mask], x_true[x_mask])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(self.trn_loader)
        epoch_metric = epoch_loss
        return epoch_loss, epoch_metric

    def _run_epoch(
        self, 
    ):
        total_loss = 0
        y_preds = np.array([])
        ys = np.array([])
        for i, (x, y) in enumerate(self.trn_loader):
            x, y = x.cuda(self.gpu_id), y.cuda(self.gpu_id)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            y_pred = torch.argmax(output, dim=1)
            y_preds = np.concatenate([y_preds, y_pred.cpu().numpy()])
            ys = np.concatenate([ys, y.cpu().numpy()])
        epoch_loss = total_loss / len(self.trn_loader)
        epoch_bacc = balanced_accuracy_score(ys, y_preds)
        return epoch_loss, epoch_bacc

    def _survival_run_epoch(self):
        for i, (x, y) in enumerate(self.trn_loader):
            x, y, e, t = x.cuda(self.gpu_id), y[0].cuda(self.gpu_id), y[1].cuda(self.gpu_id), y[2].cuda(self.gpu_id)
            self.optimizer.zero_grad()
            output = self.model(x).sigmoid() # NOTE
            loss = self._survival_loss(output, y, e)
            loss.backward()
            self.optimizer.step()
        epoch_loss, epoch_cindex = self._survival_evaluate(self.trn_loader)
        return epoch_loss, epoch_cindex

    def _pretrain_evaluate(self, data_loader: DataLoader):
        self.model.eval()
        total_loss = 0
        for i, x in enumerate(data_loader):
            x_masked, x_true, x_mask = x[0].cuda(self.gpu_id), x[1].cuda(self.gpu_id), x[2].cuda(self.gpu_id)
            x_pred = self.model(x_masked)
            total_loss += self.loss_fn(x_pred[x_mask], x_true[x_mask]).item()
        epoch_loss = total_loss / len(data_loader)
        epoch_metric = epoch_loss
        return epoch_loss, epoch_metric

    def _evaluate(self, data_loader: DataLoader):
        self.model.eval()
        y_preds = np.array([])
        ys = np.array([])
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                x, y = batch
                x, y = x.cuda(self.gpu_id), y.cuda(self.gpu_id)
                output = self.model(x) #.squeeze()
                total_loss += self.loss_fn(output, y).item()
                # TODO change accordingly
                # y_pred = output.sigmoid().round() # for binary classfication
                y_pred = torch.argmax(output, dim=1)
                y_preds = np.concatenate([y_preds, y_pred.cpu().numpy()])
                ys = np.concatenate([ys, y.cpu().numpy()])
        epoch_loss = total_loss / len(data_loader)
        epoch_bacc = balanced_accuracy_score(ys, y_preds)
        return epoch_loss, epoch_bacc

    # TODO ref ... MODIFY. use all h_{ij} and figure out ..
    def _survival_loss(
        self,
        H: torch.Tensor,
        y: torch.Tensor,
        e: torch.Tensor
    ):
        r""" Loss function for survival hazard prediction.

        # TODO some h in H is not used in the loss function, will this cause a problem? 
        # TODO may need double check , and plot the predicted KM curve to see if it makes sense
        
        Args:
            H: (batch_size, n_intervals). The H matrix, where H_{ij} is
                the hazard rate for the i-th sample at the j-th interval.
            y: (batch_size,). The index of the interval where 
                1) for uncensored patient: the event occurs.
                2) for censored patient: last survived interval index.
            e: (batch_size,). e==1 means event.
        Returns:
            loss: torch.Tensor
        """
        T = H.shape[1] # number of intervals

        intervals = torch.arange(T, device=y.device).expand_as(H)

        mask1 = (e == 1).unsqueeze(1) & (y.unsqueeze(1) == intervals) # uncensored
        mask2 = (e == 1).unsqueeze(1) & (y.unsqueeze(1) > intervals) # uncensored
        mask3 = (e == 0).unsqueeze(1) & (y.unsqueeze(1) >= intervals) # censored

        loss = - H[mask1].log().sum() - (1 - H[mask2]).log().sum() - (1 - H[mask3]).log().sum()
        loss /= (mask1.sum() + mask2.sum() + mask3.sum())

        return loss

    def _compute_cindex(
        self,
        H: np.ndarray,
        t: np.ndarray,
        e: np.ndarray
    ):
        r""" Prior knowledge about time and event are used for computing c-index
        even for validation data. NOTE
        
        """
        # compute risk scores
        r = 1 - (1-H).prod(axis=1)
        # c-index
        cindex = concordance_index_censored(
            event_indicator=e.astype(bool),
            event_time=t,
            estimate=r
        )[0]
        return cindex

    def _survival_evaluate(self, data_loader: DataLoader):
        self.model.eval()
        outputs = torch.zeros(self.n_classes).cuda(self.gpu_id)
        ys = torch.tensor([]).cuda(self.gpu_id)
        es = torch.tensor([]).cuda(self.gpu_id)
        ts = torch.tensor([]).cuda(self.gpu_id)
        with torch.no_grad():
            for i, (x, y) in enumerate(data_loader):
                x = x.cuda(self.gpu_id)
                y, e, t = y[0].cuda(self.gpu_id), y[1].cuda(self.gpu_id), y[2].cuda(self.gpu_id)
                output = self.model(x).sigmoid() # NOTE
                outputs = torch.vstack([outputs, output])
                ys = torch.hstack([ys, y])
                es = torch.hstack([es, e])
                ts = torch.hstack([ts, t])
        outputs = outputs[1:] # NOTE. remove the initial zero vector

        epoch_loss = self._survival_loss(outputs, ys, es)
        epoch_cindex = self._compute_cindex(
            outputs.cpu().numpy(),
            ts.cpu().numpy(), 
            es.cpu().numpy())
        return epoch_loss, epoch_cindex

    def train(
        self,
        max_epoch: int = 5,
        patience: int = 20 # TODO
    ):
        # TODO in fact the aggr avg operation is not entirely accurate for certain metrics

        # training
        if dist.get_rank() == 0:
            logging.info('Start training...')

        best_metric = 0
        if self.task == 'pretrain':
            best_metric = np.inf # TODO none??
        else:
            best_metric = 0

        early_stopping = 0

        trn_metrics = []
        val_metrics = []
        trn_losses = []
        val_losses = []
        for epoch in range(max_epoch):
            if dist.get_rank() == 0:
                logging.info(f'========== Epoch: {epoch + 1} ==========')
            
            self.model.train()
        
            if isinstance(self.trn_loader.sampler, DistributedSampler):
                self.trn_loader.sampler.set_epoch(epoch)
            if self.task == 'pretrain':
                epoch_loss, epoch_metric = self._pretrain_run_epoch()
            elif self.task in ['TCGA-project', 'metastasis', 'stage-classification']:
                epoch_loss, epoch_metric = self._run_epoch()
            elif self.task == 'survival':
                epoch_loss, epoch_metric = self._survival_run_epoch()

            self.scheduler.step() if self.scheduler is not None else None
        
            ## for training data
            epoch_loss = self.aggr_across_gpus(epoch_loss, op='avg')
            epoch_metric = self.aggr_across_gpus(epoch_metric, op='avg')
            if dist.get_rank() == 0:
                if self.scheduler is not None:
                    logging.info(f"Learning rate: {self.scheduler.get_last_lr()[0]}")
                logging.info(f"Train loss: {epoch_loss}")
                trn_losses.append(epoch_loss)
                logging.info(f"Train {self.task_metric}: {epoch_metric}")
                trn_metrics.append(epoch_metric)

            ## evaluation for validation data 
            if self.task == 'pretrain':
                epoch_loss, epoch_metric = self._pretrain_evaluate(self.val_loader)
            elif self.task in ['TCGA-project', 'metastasis', 'stage-classification']:
                epoch_loss, epoch_metric = self._evaluate(self.val_loader)
            elif self.task == 'survival':
                epoch_loss, epoch_metric = self._survival_evaluate(self.val_loader)
            epoch_loss = self.aggr_across_gpus(epoch_loss, op='avg')
            epoch_metric = self.aggr_across_gpus(epoch_metric, op='avg')
            if dist.get_rank() == 0:
                logging.info(f"Val loss: {epoch_loss}")
                logging.info(f"Val {self.task_metric}: {epoch_metric}\n")
                val_losses.append(epoch_loss)
                val_metrics.append(epoch_metric)

                if self.task == 'pretrain':
                    if epoch_metric <= best_metric:
                        torch.save(
                            self.model.module.module1.token_emb_vocab.weight.detach(),
                            SAVE_PATH + f'/pretrained_token_emb_weight_{self.modality}.pt')
                        torch.save(
                            self.model.module.state_dict(), 
                            SAVE_PATH + f'/model_dict_{self.task}_{self.modality}.pt')
                    if epoch_metric < best_metric:
                        best_metric = epoch_metric
                        early_stopping = 0
                    else:
                        early_stopping += 1
                elif self.task in ['TCGA-project', 'metastasis', 'stage-classification', 'survival']:
                    if epoch_metric >= best_metric:
                        torch.save(
                            self.model.module.state_dict(), 
                            SAVE_PATH + f'/model_dict_{self.task}_{self.modality}_{self.with_pretrained}_{self.tcga_subset}.pt')
                    if epoch_metric > best_metric:
                        best_metric = epoch_metric
                        early_stopping = 0
                    else:
                        early_stopping += 1
        
            dist.barrier() # TODO the all_reduce op seems doesn't correctly operates this sub-functionality? so I explicitly synchronize here
            best_metric = self.aggr_across_gpus(best_metric, op=self.metric_op)
            
            # TODO THIS IS IN FACT USELESS. DIFF PROCESSES HAVE DIFF. EARLY STOPPING VALUES.
            dist.broadcast(
                torch.tensor(early_stopping).cuda(self.gpu_id),
                src=self.default_ddp_gpu_id)
            print(f"Process {dist.get_rank()} early stopping = ", early_stopping) # TODO DEL
            
            if early_stopping == patience:
                if dist.get_rank() == 0:
                    print()
                    logging.info(f"Early stopping triggered at epoch {epoch + 1}.")
                break
        
        print(f"Process {dist.get_rank()} has exited the training loop.") # TODO DEL

        if dist.get_rank() == 0:
            _, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].plot(trn_losses, label='train')
            axes[0].plot(val_losses, label='val')
            axes[0].set_ylim(0, np.max([np.max(val_losses), np.max(trn_losses)]))
            axes[0].set_title('loss')
            axes[0].legend()
            axes[1].plot(trn_metrics, label='train')
            axes[1].plot(val_metrics, label='val')
            axes[1].set_ylim(0, np.max([np.max(val_metrics), np.max(trn_metrics)]) if self.metric_op == 'max' else np.min([np.min(val_metrics), np.min(trn_metrics)]))
            axes[1].set_title(self.task_metric)
            axes[1].legend()
            axes[1].text(0.5, 0.5, f'Best val {self.task_metric}: {best_metric:.3f}', 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=axes[1].transAxes)
            plt.tight_layout()
            plt.savefig(SAVE_PATH + f'/training_plots/{self.task}_{self.modality}_{self.with_pretrained}_{self.tcga_subset}.png')
            logging.info(f'Training complete. Best val {self.task_metric}: {best_metric:.3f}.\n')
        # TODO why stuck here??

###############################################################################
# main
###############################################################################

def main(
    rank: int,
    ddp: bool,
    world_size: int,
    max_epoch: int,
    n_heads: int,
    n_layers: int,
    batch_size: int,
    task: str,
    load_pretrained: bool,
    pretrained_emb_path: str,
    lr: float = 1e-3,
    modality: str = 'mRNA+miRNA',
    tcga_subset: str = None
):

    ## data
    if ddp:
        ddp_setup(rank, world_size)

    tensor_dtype = torch.float32 # TODO?
    trn_loader, val_loader = prep_data(
        task,
        batch_size,
        val_data=True,
        use_sampler=ddp,
        tensor_dtype=tensor_dtype,
        modality=modality,
        tcga_subset=tcga_subset
    )
    n_tokens = trn_loader.dataset.X.shape[1] if isinstance(trn_loader.dataset.X, torch.Tensor) else trn_loader.dataset.X[0].shape[1]
    n_classes = None
    if trn_loader.dataset.y is not None:
        n_classes = torch.unique(trn_loader.dataset.y).shape[0] if isinstance(trn_loader.dataset.y, torch.Tensor) else torch.unique(trn_loader.dataset.y[0]).shape[0]
        if (task == 'survival') & (-1 in trn_loader.dataset.y[0]):
            n_classes -= 1

    ## model
    pretrain = True if task == 'pretrain' else False # TODO
    assert not ((pretrain) & (load_pretrained)), "Cannot load pretrained model when pretraining."
    pretrained_token_emb_weight = None
    if load_pretrained:
        if rank == 0:
            logging.info("Loading pretrained token embedding...")
        # TODO NOTE CHOOSE. DEL RANDOM EMBEDDING FOR SANITY CHECK
        pretrained_emb_path = SAVE_PATH + f'/pretrained_token_emb_weight_{modality}.pt'
        # pretrained_emb_path = SAVE_PATH + f'/random_pretrained_token_emb_weight_{modality}.pt'
        pretrained_token_emb_weight = torch.load(pretrained_emb_path)
    model = MMBL(
        n_tokens = n_tokens,
        n_classes = n_classes,
        d_e = EMB_DIM,
        n_heads = n_heads,
        n_encoders = n_layers,
        dropout = DROP_RATE,
        pretrained_token_emb_weight=pretrained_token_emb_weight, # TODO
        tensor_dtype=tensor_dtype,
        pretrain = pretrain # NOTE
    )
    model.cuda(rank) # NOTE use GPU
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    ## train
    ### loss function TODO
    loss_fn = None
    if task=='pretrain':
        loss_fn = torch.nn.MSELoss()
    elif task in ['TCGA-project', 'stage-classification',  'metastasis']:
        loss_fn = torch.nn.CrossEntropyLoss()
    # elif task in ['metastasis']:
    #     loss_fn = torch.nn.BCEWithLogitsLoss()
    ### optimizer & scheduler
    scheduler = None
    if task == 'pretrain':
        if dist.get_rank() == 0:
            logging.info("Using AdamW optimizer and CosineAnnealingLR scheduler.")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr, 
            betas=(0.9, 0.95), 
            eps=1e-8,
            weight_decay=1e-2)
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, 
        #     step_size=20, # once every step_size epochs
        #     gamma=0.1 # decay rate
        # )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=20, # TODO
            eta_min=0.1 * lr)
    elif task in ['TCGA-project', 'stage-classification', 'metastasis', 'survival', 'survival-binary']:
        if dist.get_rank() == 0:
            logging.info("Using Adam optimizer.")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr, 
            weight_decay=1e-3)

    trainer = Trainer(
        task = task,
        modality = modality,
        model = model,
        optimizer = optimizer,
        scheduler=scheduler, # TODO
        trn_loader = trn_loader,
        val_loader = val_loader,
        loss_func = loss_fn,
        gpu_id = rank,
        tcga_subset = tcga_subset
    )
    trainer.train(max_epoch = max_epoch)

    if ddp > 1:
        dist.barrier()
        dist.destroy_process_group() # NOTE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddp', action='store_true',
                        help='Data Distributed Paralell for training on multiple GPUs.')
    parser.add_argument('--max_epoch', type=int, default=MAX_EPOCH)
    parser.add_argument('--n_heads', type=int, default=HEADS)
    parser.add_argument('--n_layers', type=int, default=LAYERS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--task', type=str)
    parser.add_argument('--load_pretrained', action='store_true', default=False)
    parser.add_argument('--pretrained_emb_path', type=str, 
                        default=f'./saved/pretrained_token_emb_weight.pt') # TODO NOTE might need modifications
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--modality', type=str, default='mRNA+miRNA')
    parser.add_argument('--tcga_subset', type=str)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    assert args.task in ['pretrain', 'TCGA-project', 'metastasis', 
                         'stage-classification',  
                         'survival'], \
        "Task must be correctly specified."

    set_random_seed(args.random_seed)

    world_size = None

    if args.ddp:
        try:
            world_size = torch.cuda.device_count()
        except:
            raise ValueError('CUDA not available.')
        assert world_size > 1, "Single GPU cannot perform DDP."
        logging.info(f"Task is {args.task}. Using DDP, world size = {world_size}.")

        mp.spawn(
            main,
            args=(
                args.ddp,
                world_size,
                args.max_epoch,
                args.n_heads,
                args.n_layers,
                args.batch_size,
                args.task,
                args.load_pretrained,
                args.pretrained_emb_path,
                args.lr,
                args.modality,
                args.tcga_subset
            ),
            nprocs=world_size
        )
    else:
        logging.info(f"Task is {args.task}. Not using DDP.")

        assert world_size == 1
        main(
            rank=0,
            ddp=args.ddp,
            world_size=1,
            max_epoch=args.max_epoch,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            batch_size=args.batch_size,
            task=args.task,
            load_pretrained=args.load_pretrained,
            pretrained_emb_path=args.pretrained_emb_path,
            lr=args.lr,
            modality=args.modality,
            tcga_subset=args.tcga_subset
        )

# python v8_pan_cancer_model_training.py --ddp --task pretrain --modality mRNA+miRNA --max_epoch 5 --lr 0.0003