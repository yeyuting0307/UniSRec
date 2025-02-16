#%%
import random
import numpy as np
import torch
from recbole.sampler.sampler import SeqSampler, Sampler

class BasicSeqSampler(SeqSampler):
    def __init__(self, dataset, distribution="uniform", alpha=1.0):
        super().__init__(
            dataset = dataset, 
            distribution=distribution, 
            alpha = alpha
        )

    def sample_by_user_ids(self, user_ids, item_ids, num):
        neg_item_ids = []
        for i, uid in enumerate(user_ids):
            choices = list(set(item_ids) - set([item_ids[i]]))
            neg_item_ids.extend(random.sample(choices, 1))
        return neg_item_ids   
        
        