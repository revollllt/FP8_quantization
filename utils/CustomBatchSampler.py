import random

class CustomBatchSampler:
    def __init__(self, dataloader, num_batches, random_sample=False, start_index=0, step=1):
        self.dataloader = dataloader
        self.num_batches = num_batches
        self.random_sample = random_sample
        self.start_index = start_index
        self.step = step
        self.total_batches = len(dataloader)

    def __iter__(self):
        if self.random_sample:
            # 随机采样
            indices = random.sample(range(self.total_batches), min(self.num_batches, self.total_batches))
            for idx in sorted(indices):
                for i, batch in enumerate(self.dataloader):
                    if i == idx:
                        yield batch
                        break
        else:
            # 顺序采样
            for i, batch in enumerate(self.dataloader):
                if i < self.start_index:
                    continue
                if (i - self.start_index) % self.step == 0:
                    yield batch
                    if (i - self.start_index) // self.step + 1 >= self.num_batches:
                        break

    def __len__(self):
        return min(self.num_batches, self.total_batches)