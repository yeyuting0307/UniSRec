#%%
import os
import torch
import random
import pandas as pd
import numpy as np
from recbole.data.dataset import SequentialDataset
from recbole.data.interaction import Interaction
from recbole.data.dataloader.general_dataloader import TrainDataLoader
from recbole.utils import FeatureSource, FeatureType

class UniSRecDataset(SequentialDataset):
    def __init__(self, config, **kwargs):
        super(UniSRecDataset, self).__init__(config, **kwargs)
        self.item_text_file_path = os.path.join(
            os.path.dirname(__file__), 
            os.pardir,
            config['item_text_csv_path']
        )
        self.item_text = self.load_item_text(self.item_text_file_path)

    def _remap(self, remap_list):
        if len(remap_list) == 0:
            return
        tokens, split_point = self._concat_remaped_tokens(remap_list)

        new_ids_list, mp = np.array([int(tok) for tok in tokens]), tokens
        new_ids_list = np.split(new_ids_list, split_point)
        
        token_id = {t: new_ids_list[0][i] for i, t in enumerate(mp)}
        mp = np.array(["[PAD]"] + list(mp))
        token_id.update({"[PAD]": 0})

        for (feat, field, ftype), new_ids in zip(remap_list, new_ids_list):
            if field not in self.field2id_token:
                self.field2id_token[field] = mp
                self.field2token_id[field] = token_id
            if ftype == FeatureType.TOKEN:
                feat[field] = new_ids
            elif ftype == FeatureType.TOKEN_SEQ:
                split_point = np.cumsum(feat[field].agg(len))[:-1]
                feat[field] = np.split(new_ids, split_point)

    def load_item_text(self, path):
        df_item_text = pd.read_csv(path)
        item_text = df_item_text\
            [['item_token', 'item_text']]\
            .set_index('item_token')\
            .to_dict('dict')\
            .get('item_text')
        item_text[0] = "[PAD]"
        return item_text
        
class FinetuneDataLoader(TrainDataLoader):
    def __init__(self, config, dataset, sampler=None, shuffle=True):
        super(FinetuneDataLoader, self).__init__(config, dataset, sampler, shuffle)
        self.transform = DataTransformer(config)

    def _next_batch_data(self):
        cur_data = super()._next_batch_data()
        transformed_data = self.transform(self, cur_data[0])
        return (transformed_data, *cur_data[1:])

class PretrainUniSRecDataset(UniSRecDataset):
    def __init__(self, config, **kwargs):
        super(PretrainUniSRecDataset, self).__init__(config, **kwargs)
        
class PretrainDataLoader(TrainDataLoader):
    def __init__(self, config, dataset, sampler=None, shuffle=True):
        super(PretrainDataLoader, self).__init__(config, dataset, sampler, shuffle)
        self.transform = DataTransformer(config)

    def _next_batch_data(self):
        cur_data = super()._next_batch_data()
        transformed_data = self.transform(self, cur_data[0])
        return (transformed_data, *cur_data[1:])

class DataTransformer:
    def __init__(self, config):
        self.item_seq_mask_prob = config['ITEM_SEQ_MASK_PROB']
        self.item_seq_mask_ratio = config['ITEM_SEQ_MASK_RATIO']

    def __call__(self,  dataset, interaction):
        item_seq = interaction['item_id_list']
        item_seq_len = interaction['item_id_list_len']

        mask_p = torch.full_like(item_seq, self.item_seq_mask_ratio, dtype=torch.float)
        is_mask = torch.bernoulli(mask_p).to(torch.bool)

        item_seq_aug = item_seq
        item_seq_len_aug = item_seq_len

        rand = random.random()
        if rand < self.item_seq_mask_prob:
            seq_mask = item_seq.eq(0).to(torch.bool)
            mask = torch.logical_or(is_mask, seq_mask)
            mask[:,0] = True 
            drop_index = torch.cumsum(mask, dim=1) - 1
            item_seq_aug = torch.zeros_like(item_seq).scatter(dim=-1, index=drop_index, src=item_seq)
            item_seq_len_aug = torch.gather(drop_index, 1, (item_seq_len - 1).unsqueeze(1)).squeeze() + 1
        
        interaction.update(Interaction({
            'item_id_list_aug': item_seq_aug,
            'item_id_list_len_aug': item_seq_len_aug,
        }))

        return interaction

#%%
if __name__ == "__main__":
    from recbole.config import Config
    from models.unisrec import UniSRec
    dir_path = os.path.join(os.path.dirname(__file__)) 
    config = Config(
            model=UniSRec, 
            dataset="project", 
            config_file_list = [
                "configs/Sequential.yaml",
                "configs/SASRec.yaml",
                "configs/UniSRec.yaml",
                "configs/inference.yaml",
            ], 
    )
    config['item_text_csv_path'] = "data/fixed/project/item_token_text.csv"
    config["data_path"] = "data/inference/project/"

    dataset = UniSRecDataset(config)
    test_data = dataset.build()[0]

    print(test_data['user_id'])
    print(test_data['item_id_list'])
    print(test_data['item_id'])

