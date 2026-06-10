import torch
from torch.utils.data import DataLoader
from dataloader.sampler import FixedDistributedSampler
import numpy as np
import random
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
class FixedLipLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, world_size=1, batch_size=2, 
                 shuffle=True, num_workers=8, drop_last=True,worker_init_fn = seed_worker):
        sampler = FixedDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            shuffle=shuffle
        )
        
        super().__init__(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=self._collate,
            worker_init_fn = seed_worker
        )
    def _collate(self, batch):
        # return {
        #     'target': torch.stack([x['target'] for x in batch]),
        #     'inter': torch.stack([x['inter'] for x in batch]),
        #     'noise': torch.stack([x['noise'] for x in batch]),
        #     'lip_0': torch.stack([x['lip_0'] for x in batch]),
        #     'lip_1': torch.stack([x['lip_1'] for x in batch])
        # }
    #     speaker_data = torch.stack([
    #     # 通道维度合并（维度工程优化）
    #     torch.stack([x['target'], x['inter']], dim=0).permute(0, 1)  # [2, T] -> 保持时间连续性
    #     for x in batch
    # ], dim=0)
        """由于长度固定，直接堆叠"""
        return {
        # 'padded_speaker': speaker_data,  # [B, 2, T]
        'mix': torch.stack([x['mix'] for x in batch]),
        'tar': torch.stack([x['tar'] for x in batch]),
        'noise': torch.stack([x['noise'] for x in batch]),
        # 多模态数据对齐（扩展性考虑）
        'lip_0': torch.stack([x['lip_0'] for x in batch]),
        'lip_1': torch.stack([x['lip_1'] for x in batch]),
        'lip_2': torch.stack([x['lip_2'] for x in batch]),
        'view_mask' : torch.stack([x['view_mask'] for x in batch]),
        'ilens': torch.stack([x['ilens'] for x in batch])
    }

# class TruncateDataLoader(DataLoader):
#     def __init__(self, dataset, world_size, trun_range=[64000, 64000], step=1,
#                  shuffle=False, batch_size=2, num_workers=1, drop_last=False):
#         self.dataset = dataset
#         self.batch_sampler = TrunBatchSampler(self.dataset,
#                                               world_size,
#                                               trun_range=trun_range,
#                                               step=step,
#                                               shuffle=shuffle,
#                                               batch_size=batch_size,
#                                               drop_last=drop_last)
#         super().__init__(self.dataset,
#                          collate_fn=self.collate_fn,
#                          batch_sampler=self.batch_sampler,
#                          num_workers=num_workers,
#                          pin_memory=True)


#     def collate_fn(self, batch):
#         stacked = {
#             'noise': torch.stack([x['noise'] for x in batch]),
#             'target': torch.stack([x['target'] for x in batch]),
#             'inter': torch.stack([x['inter'] for x in batch]),
#             'lip_0': torch.stack([x['lip_0'] for x in batch]),
#             'lip_1': torch.stack([x['lip_1'] for x in batch])
#             # 'has_inter': torch.stack([x['has_inter'] for x in batch])
#         }
#         return stacked

class InfDataLoader(DataLoader):
    def __init__(self, dataset, trun_range=[64000, 64000], step=1,
                 shuffle=False, batch_size=1, num_workers=1, drop_last=False):
        self.dataset = dataset
        self.batch_sampler = SeqBatchSampler(self.dataset,
                                              trun_range=trun_range,
                                              step=step,
                                              shuffle=shuffle,
                                              batch_size=batch_size,
                                              drop_last=drop_last)
        super().__init__(self.dataset,
                         collate_fn=self.collate_fn,
                         batch_sampler=self.batch_sampler,
                         num_workers=num_workers,
                         pin_memory=True)


    def collate_fn(self, batch):
        batch = list(zip(*batch))
        utt_list, mix_wav, ilens, lip_emb = batch
        mix_wav = torch.stack(mix_wav, dim=0)
        ilens = torch.stack(ilens, dim=0)
        lip_emb = torch.stack(lip_emb, dim=0)
        #s1_wav = torch.stack(s1_wav, dim=0)
        #s1_wav = torch.stack(s2_wav, dim=0)
        return utt_list, mix_wav, ilens, lip_emb