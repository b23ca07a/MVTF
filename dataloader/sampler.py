import torch
from torch.utils.data import RandomSampler, SequentialSampler
import random

from torch.utils.data.distributed import DistributedSampler
import math

import torch
import math
import random
from torch.utils.data.distributed import DistributedSampler

class FixedDistributedSampler(DistributedSampler):
    """支持多数据源对齐的分布式采样器"""
    def __iter__(self):
        indices = list(super().__iter__())
        
        # 诊断日志：记录每个进程分配的样本数
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            print(f"[Rank {rank}] 分配样本数: {len(indices)}")
        actual_len = len(self.dataset)
        # if actual_len != 10000:
        #     print(f"警告: 数据集实际长度={actual_len}, 预期10000")
        # 生成噪声索引（确保索引在噪声列表范围内）
        noise_indices = [i % len(self.dataset.noise_list) for i in indices]
        
        # 返回配对索引
        return zip(indices, noise_indices)
# class FixedDistributedSampler(DistributedSampler):
#     def __iter__(self):
#         # 非分布式环境下返回完整数据集
#         if not torch.distributed.is_initialized():
#             print("this error???")
#             total_size = len(self.dataset)
#             indices = list(range(total_size))
            
#             # 添加此部分确保长度匹配
#             if not self.drop_last:
#                 # 添加额外样本使分区大小相等
#                 padding_size = total_size - len(indices)
#                 if padding_size <= len(indices):
#                     indices += indices[:padding_size]
#                 else:
#                     indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
#             else:
#                 # 删除尾部数据使可整除
#                 indices = indices[:total_size - total_size % self.num_replicas]
            
#             # 生成噪声索引
#             noise_indices = (i % len(self.dataset.noise_list) for i in indices)
#             return zip(indices, noise_indices)
        
#         # 分布式环境保持原逻辑
#         indices = super().__iter__()
#         noise_indices = (i % len(self.dataset.noise_list) for i in indices)
#         return zip(indices, noise_indices)
# class FixedDistributedSampler(DistributedSampler):
#     """支持多数据源对齐的分布式采样器"""
#     def __iter__(self):
#         indices = super().__iter__()
#         # 噪声索引循环模式
#         noise_indices = (i % len(self.dataset.noise_list) for i in indices)
#         return zip(indices, noise_indices)
    
# class TripleDistributedSampler(DistributedSampler):
#     """支持三数据源的分布式采样器（主数据/干扰数据/噪声）"""
#     def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
#         super().__init__(dataset, num_replicas, rank, shuffle)
#         self.dataset = dataset
#         self.epoch = 0
#         self.seed = seed

#     def _generate_indices(self, data_len, generator):
#         """生成分布式安全的索引序列"""
#         indices = torch.randperm(data_len, generator=generator).tolist()
        
#         # 填充或截断以适配分布式训练
#         if not self.drop_last:
#             padding_size = self.total_size - len(indices)
#             if padding_size > 0:
#                 indices += indices[:padding_size]
#         else:
#             indices = indices[:self.total_size]
        
#         # 子采样当前进程的数据
#         return indices[self.rank:self.total_size:self.num_replicas]

#     def __iter__(self):
#         # 同步随机种子
#         g = torch.Generator()
#         g.manual_seed(self.seed + self.epoch)
        
#         # 生成主数据索引
#         main_indices = self._generate_indices(len(self.dataset), g)
        
#         # 生成干扰数据索引
#         g_inter = torch.Generator()
#         g_inter.manual_seed(self.seed + self.epoch + 1)
#         inter_indices = self._generate_indices(self.dataset.inter_len, g_inter)
        
#         # 生成噪声数据索引
#         g_noise = torch.Generator()
#         g_noise.manual_seed(self.seed + self.epoch + 2)
#         noise_indices = self._generate_indices(self.dataset.noise_len, g_noise)
        
#         # 组合索引并截断到最小长度
#         min_len = min(len(main_indices), len(inter_indices), len(noise_indices))
#         return iter(zip(
#             main_indices[:min_len],
#             inter_indices[:min_len],
#             noise_indices[:min_len]
#         ))

#     def set_epoch(self, epoch):
#         self.epoch = epoch

# class EnhancedTrunBatchSampler:
#     """支持动态截断长度的分布式批次采样器"""
#     def __init__(self, dataset, world_size, trun_range, step=1, 
#                  shuffle=True, batch_size=32, drop_last=False):
#         assert len(trun_range) == 2, "截断范围应为[start, end]"
#         self.trun_lens = list(range(trun_range[0], trun_range[1]+1, step))
#         self.batch_size = batch_size
#         self.drop_last = drop_last
        
#         # 初始化分布式采样器
#         self.sampler = TripleDistributedSampler(
#             dataset,
#             num_replicas=world_size,
#             shuffle=shuffle,
#             seed=hash(dataset) % 2**32
#         )
        
#         # 预生成截断长度序列以确保跨进程一致性
#         self.length_generator = torch.Generator()
#         self.length_generator.manual_seed(hash(dataset) % 2**32)

#     def __iter__(self):
#         batch = []
#         current_length = None
        
#         # 同步随机状态
#         self.length_generator.manual_seed(
#             self.sampler.seed + self.sampler.epoch * 1000
#         )
        
#         for indices in self.sampler:
#             # 为当前批次生成统一截断长度
#             if current_length is None or len(batch) == 0:
#                 current_length = self.trun_lens[
#                     torch.randint(0, len(self.trun_lens), 
#                     (1,), 
#                     generator=self.length_generator
#                 ).item()]
            
#             batch.append((*indices, current_length))
            
#             if len(batch) == self.batch_size:
#                 yield batch
#                 batch = []
#                 current_length = None
        
#         if len(batch) > 0 and not self.drop_last:
#             yield batch

#     def __len__(self):
#         if self.drop_last:
#             return len(self.sampler) // self.batch_size
#         return (len(self.sampler) + self.batch_size - 1) // self.batch_size

#     def set_epoch(self, epoch):
#         self.sampler.set_epoch(epoch)

# class DoubleDistributedSampler(DistributedSampler):
#     def __init__(self, dataset, num_replicas=0, shuffle=True):
#         super().__init__(dataset, num_replicas=num_replicas, shuffle=shuffle)
    
#     def __iter__(self):
#         if self.shuffle:
#             # deterministically shuffle based on epoch and seed
#             g = torch.Generator()
#             g.manual_seed(self.seed + self.epoch)
#             indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
#         else:
#             indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

#         if not self.drop_last:
#             # add extra samples to make it evenly divisible
#             padding_size = self.total_size - len(indices)
#             if padding_size <= len(indices):
#                 indices += indices[:padding_size]
#             else:
#                 indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
#         else:
#             # remove tail of data to make it evenly divisible.
#             indices = indices[:self.total_size]
#         assert len(indices) == self.total_size

#         # subsample
#         indices = indices[self.rank:self.total_size:self.num_replicas]
#         assert len(indices) == self.num_samples
        
#         if self.shuffle:
#             # deterministically shuffle based on epoch and seed
#             g2 = torch.Generator()
#             g2.manual_seed(self.seed + 1 + self.epoch)
#             indices2 = torch.randperm(len(self.dataset), generator=g2).tolist()  # type: ignore[arg-type]
#         else:
#             indices2 = list(range(len(self.dataset)))  # type: ignore[arg-type]

#         if not self.drop_last:
#             # add extra samples to make it evenly divisible
#             padding_size = self.total_size - len(indices2)
#             if padding_size <= len(indices2):
#                 indices2 += indices2[:padding_size]
#             else:
#                 indices2 += (indices2 * math.ceil(padding_size / len(indices2)))[:padding_size]
#         else:
#             # remove tail of data to make it evenly divisible.
#             indices2 = indices2[:self.total_size]
#         assert len(indices2) == self.total_size

#         # subsample
#         indices2 = indices2[self.rank:self.total_size:self.num_replicas]
#         assert len(indices2) == self.num_samples

#         return iter(zip(indices, indices2))

# class TripleDistributedSampler(DistributedSampler):
#     def __init__(self, dataset, num_replicas=0, shuffle=True):
#         super().__init__(dataset, num_replicas=num_replicas, shuffle=shuffle)
    
#     def __iter__(self):
        
#         if self.shuffle:
#             # deterministically shuffle based on epoch and seed
#             g = torch.Generator()
#             g.manual_seed(self.seed + self.epoch)
#             indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
#         else:
#             indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

#         if not self.drop_last:
#             # add extra samples to make it evenly divisible
#             padding_size = self.total_size - len(indices)
#             if padding_size <= len(indices):
#                 indices += indices[:padding_size]
#             else:
#                 indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
#         else:
#             # remove tail of data to make it evenly divisible.
#             indices = indices[:self.total_size]
#         assert len(indices) == self.total_size
        
#         # subsample
#         indices = indices[self.rank:self.total_size:self.num_replicas]
#         assert len(indices) == self.num_samples
        
#         # Consistent in all processes
#         if self.shuffle:
#             # When total_size > inter_len, traverse inter_len multiple times to match total_size
#             g2 = torch.Generator()
#             g2.manual_seed(self.seed + 1 + self.epoch)
#             res = torch.tensor(random.sample(range(self.dataset.inter_len), int(self.total_size % self.dataset.inter_len)), dtype=int)
#             tmp = [torch.arange(self.dataset.inter_len) for i in range(int(self.total_size / self.dataset.inter_len))]
#             total = torch.cat(tmp + [res])
#             indices2 = total[torch.randperm(self.total_size, generator=g2)].tolist()  # type: ignore[arg-type]
#         else:
#             indices2 = list(range(len(self.dataset)))  # To be edited
#         assert len(indices2) == self.total_size
        
#         # subsample
#         indices2 = indices2[self.rank:self.total_size:self.num_replicas]
#         assert len(indices2) == self.num_samples
        
#         if self.shuffle:
#             # When total_size > noise_len, traverse noise_len multiple times to match total_size
#             g3 = torch.Generator()
#             g3.manual_seed(self.seed + 2 + self.epoch)
#             res = torch.tensor(random.sample(range(self.dataset.noise_len), int(self.total_size % self.dataset.noise_len)))
#             tmp = [torch.arange(self.dataset.noise_len) for i in range(int(self.total_size / self.dataset.noise_len))]
#             total = torch.cat(tmp + [res])
#             noise_indices = total[torch.randperm(self.total_size, generator=g3)].tolist()  # type: ignore[arg-type]
#         else:
#             noise_indices = list(range(len(self.dataset)))  # To be edited
#         assert len(noise_indices) == self.total_size
        
#         # subsample
#         noise_indices = noise_indices[self.rank:self.total_size:self.num_replicas]
#         assert len(noise_indices) == self.num_samples

#         return iter(zip(zip(indices, indices2), noise_indices))

# class TrunBatchSampler(object):
#   '''
#   In case that `shuffle=False, batch_size=4`, 
#   for common BatchSampler, it generates: 
#   Batch 0: [0, 1, 2, 3]
#   Batch 1: [4, 5, 6, 7]
#   ...

#   For TrunBatchSampler, it generates:
#   Batch 0: [(0,a), (1,a), (2,a), (3,a)]
#   Batch 1: [(4,b), (5,b), (6,b), (7,b)]
#   ...
#   where a,b... are random values in trun_range.

#   The generated sequences are fed into function `__getitem__` of class Dataset.
#   It can be used for dynamic LSTM sequences or dynamic nframe-spectrogram CNN inputs.
#   ''' 
#   def __init__(self, dataset, world_size, trun_range, step=1, shuffle=False, batch_size=1, drop_last=False):
#     assert isinstance(trun_range, list) and len(trun_range) == 2
#     self.trun_lens = list(range(trun_range[0], trun_range[1]+1, step))
#     self.step = step
#     self.batch_size = batch_size
#     self.drop_last = drop_last    
#     # if shuffle:
#     #   self.sampler = RandomSampler(dataset)
#     # else:
#     #   self.sampler = TripleDistributedSampler(dataset, num_replicas=world_size, shuffle=True)
#     self.sampler = TripleDistributedSampler(dataset, num_replicas=world_size, shuffle=shuffle)

#   def __iter__(self):
#     batch = []
#     n = random.choice(self.trun_lens)
#     for (idx_comb,idx2) in self.sampler:
#       batch.append((idx_comb, idx2, n))
#       if len(batch) == self.batch_size:
#         yield batch
#         batch = []
#         n = random.choice(self.trun_lens)
#     if len(batch) > 0 and not self.drop_last:
#       yield batch

#   def __len__(self):
#     if self.drop_last:
#       return len(self.sampler) // self.batch_size
#     else:
#       return (len(self.sampler) + self.batch_size - 1) // self.batch_size

# class SeqBatchSampler(object):
#   '''
#   In case that `shuffle=False, batch_size=4`, 
#   for common BatchSampler, it generates: 
#   Batch 0: [0, 1, 2, 3]
#   Batch 1: [4, 5, 6, 7]
#   ...

#   For TrunBatchSampler, it generates:
#   Batch 0: [(0,a), (1,a), (2,a), (3,a)]
#   Batch 1: [(4,b), (5,b), (6,b), (7,b)]
#   ...
#   where a,b... are random values in trun_range.

#   The generated sequences are fed into function `__getitem__` of class Dataset.
#   It can be used for dynamic LSTM sequences or dynamic nframe-spectrogram CNN inputs.
#   ''' 
#   def __init__(self, dataset, trun_range, step=1, shuffle=False, batch_size=1, drop_last=False):
#     assert isinstance(trun_range, list) and len(trun_range) == 2
#     self.trun_lens = list(range(trun_range[0], trun_range[1]+1, step))
#     self.step = step
#     self.batch_size = batch_size
#     self.drop_last = drop_last    
#     if shuffle:
#       self.sampler = RandomSampler(dataset)
#     else:
#       # print('distributed shuffle', shuffle)
#       # self.sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=16, shuffle=shuffle)
#       self.sampler = SequentialSampler(dataset)

#   def __iter__(self):
#     batch = []
#     # n = random.choice(self.trun_lens)
#     for idx in self.sampler:
#       batch.append((idx, self.trun_lens[-1]))
#       if len(batch) == self.batch_size:
#         yield batch
#         batch = []
#         # n = random.choice(self.trun_lens)
#     if len(batch) > 0 and not self.drop_last:
#       yield batch

#   def __len__(self):
#     if self.drop_last:
#       return len(self.sampler) // self.batch_size
#     else:
#       return (len(self.sampler) + self.batch_size - 1) // self.batch_size
