#%%
###############################################################################
# trying DDP (run with command: python try_ddp.py 
###############################################################################
import os
import numpy as np 
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

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
            dist.all_reduce(-value, op=dist.ReduceOp.MAX)
            return -value.item()


WORLD_SIZE = 2 # torch.cuda.device_count()
BATCH_SIZE = 32
MAX_EPOCH = 5

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        trn_loader: DataLoader,
        gpu_id: int,
    ) -> None:
        self.optimizer = optimizer
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.trn_loader = trn_loader
        self.optimizer = optimizer
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, x, y):
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        self.optimizer.step()
    
    def train(self, max_epoch: int):
        self.model.train()
        best_metric = np.inf
        for epoch in range(max_epoch):
            self.trn_loader.sampler.set_epoch(epoch)
            for i, (x, y) in enumerate(self.trn_loader):
                if i == 0:
                    print(
                        f"GPU: {self.gpu_id},  \
                        epoch: {epoch}, \
                        batch size: {x.shape[0]}, \
                        steps: {len(self.trn_loader)}"
                    )
                x, y = x.to(self.gpu_id), y.to(self.gpu_id)
                self._run_batch(x, y)
            
            if dist.get_rank() == 0:
                for i in range(10000):
                    best_metric += 1.0/300



class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]
    
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]

trn_dataset = MyTrainDataset(2048)


# main
def main(rank: int, world_size: int, max_epoch: int, batch_size: int):
    ddp_setup(rank, world_size)

    trn_dataset = MyTrainDataset(2048)
    trn_loader = DataLoader(
        trn_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=DistributedSampler(trn_dataset),
        pin_memory=True
    )

    model = torch.nn.Linear(20, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model, 
        gpu_id=rank,
        optimizer=optimizer, 
        trn_loader=trn_loader,)
    trainer.train(max_epoch=max_epoch)

    destroy_process_group() # NOTE

if __name__ == '__main__':
    mp.spawn(main, args=(WORLD_SIZE, MAX_EPOCH, BATCH_SIZE), nprocs=WORLD_SIZE)



# #%%
# ###############################################################################
# # trying DDP (run with command: torchrun try_ddp.py 
# ###############################################################################
# import os
# import torch
# import torch.distributed as dist
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset

# import torch.multiprocessing as mp
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group

# def ddp_setup():
#     # os.environ['MASTER_ADDR'] = 'localhost'
#     # os.environ['MASTER_PORT'] = '12355'
#     dist.init_process_group("nccl")
#     torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

# BATCH_SIZE = 32
# MAX_EPOCH = 5

# class Trainer:
#     def __init__(
#         self,
#         model: torch.nn.Module,
#         optimizer: torch.optim.Optimizer,
#         trn_loader: DataLoader) -> None:

#         self.optimizer = optimizer
#         self.gpu_id = int(os.environ['LOCAL_RANK'])
#         self.model = model.to(self.gpu_id)
#         self.trn_loader = trn_loader
#         self.optimizer = optimizer
#         self.model = DDP(model, device_ids=[self.gpu_id])

#     def _run_batch(self, x, y):
#         self.optimizer.zero_grad()
#         output = self.model(x)
#         loss = F.cross_entropy(output, y)
#         loss.backward()
#         self.optimizer.step()
    
#     def train(self, max_epoch: int):
#         self.model.train()
#         for epoch in range(max_epoch):
#             self.trn_loader.sampler.set_epoch(epoch)
#             for i, (x, y) in enumerate(self.trn_loader):
#                 if i == 0:
#                     print(
#                         f"GPU: {self.gpu_id},  \
#                         epoch: {epoch}, \
#                         batch size: {x.shape[0]}, \
#                         steps: {len(self.trn_loader)}")
#                 x, y = x.to(self.gpu_id), y.to(self.gpu_id)
#                 self._run_batch(x, y)

# class MyTrainDataset(Dataset):
#     def __init__(self, size):
#         self.size = size
#         self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]
    
#     def __len__(self):
#         return self.size

#     def __getitem__(self, index):
#         return self.data[index]

# trn_dataset = MyTrainDataset(2048)


# # main
# def main(max_epoch: int, batch_size: int):
#     ddp_setup()

#     trn_dataset = MyTrainDataset(2048)
#     trn_loader = DataLoader(
#         trn_dataset, 
#         batch_size=batch_size, 
#         shuffle=False, 
#         sampler=DistributedSampler(trn_dataset),
#         pin_memory=True
#     )

#     model = torch.nn.Linear(20, 1)
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#     trainer = Trainer(
#         model=model, 
#         optimizer=optimizer, 
#         trn_loader=trn_loader)
#     trainer.train(max_epoch=max_epoch)

#     destroy_process_group() # NOTE

# if __name__ == '__main__':
#     main(MAX_EPOCH, BATCH_SIZE)