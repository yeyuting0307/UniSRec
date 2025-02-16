import os
import sys
import pandas as pd
import torch
import logging
from torch.nn import functional as F
from logging import getLogger
from recbole.config import Config
from models.unisrec import UniSRec
from datasets.unisrec import UniSRecDataset

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

def eval_hit_n(top_n_idx, true_y):
    """ Evaluate the hit rate of the top_n_idx """
    hit_rate = sum(true_y[i] in idx_list for i, idx_list in enumerate(top_n_idx)) / len(true_y)
    print(f"{sum(true_y[i] in idx_list for i, idx_list in enumerate(top_n_idx))} / {len(true_y)} = {hit_rate}")
    return hit_rate

def load_model(config, dir_path):
    """ Load the UniSRec model and its weights """
    model = UniSRec(config, UniSRecDataset(config).build()[0]).to(config['device'])
    checkpoint_path = os.path.join(dir_path, "checkpoints", "UniSRec-prior.pth")
    checkpoint = torch.load(checkpoint_path, map_location=config['device'])

    for name, param in checkpoint['state_dict'].items():
        if name.startswith('item_embedding'):
            logging.info(f"Skip unused {name}")
            continue
        model.state_dict()[name].copy_(param)

    model.load_state_dict(model.state_dict())
    return model

def prepare_dataframe(test_data, top_inx, top_v):
    """ Prepare a DataFrame for saving the prediction results """
    df_list = []
    for i in range(len(top_inx)):
        user_id = test_data['user_id'][i].tolist()
        rec_list = top_inx[i].tolist()
        val_list = top_v[i].tolist()
        rank = list(range(1, len(rec_list) + 1))
        sub_df = pd.DataFrame({
            "user_token": user_id,
            "item_token": rec_list,
            "score": val_list,
            "rank": rank
        })
        df_list.append(sub_df)
    return pd.concat(df_list)

def predict(batch_size=1024):
    """ Predict the top k items for each user """
    dir_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    config = Config(
        model=UniSRec,
        dataset="project",
        config_file_list=[
            os.path.join(dir_path, "configs", "Sequential.yaml"),
            os.path.join(dir_path, "configs", "SASRec.yaml"),
            os.path.join(dir_path, "configs", "UniSRec.yaml"),
            os.path.join(dir_path, "configs", "inference.yaml"),
        ],
    )
    config["data_path"] = os.path.join(dir_path, config["data_path"])
    logging.info(config)

    dataset = UniSRecDataset(config)
    test_data = dataset.build()[0]

    model = load_model(config, dir_path)

    predictions = []
    for i in range(len(test_data) // batch_size + 1):
        logging.info(f"Processing batch: {i * batch_size}:{(i + 1) * batch_size}")
        batch_data = test_data[i * batch_size:(i + 1) * batch_size]
        predictions.extend(model.full_sort_predict(batch_data).tolist())

    predictions = torch.tensor(predictions)
    top_k = predictions.shape[1]
    top_v, top_inx = predictions.topk(k=top_k, dim=-1)

    result_df = prepare_dataframe(test_data, top_inx, top_v)
    logging.info(f"Predicted DataFrame size: {result_df.shape}")

    return result_df

if __name__ == "__main__":
    df = predict()
