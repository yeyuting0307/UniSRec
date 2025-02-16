import os
import torch
import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, set_color
from models.unisrec import UniSRec
from datasets.unisrec import UniSRecDataset
from trainer.unisrec import FinetuneTrainer

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

def load_pretrained_weights(model, checkpoint_path, device):
    """ Load pretrained weights into the model """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = model.state_dict()

    for name, param in checkpoint['state_dict'].items():
        if name.startswith('item_embedding'):
            logging.info(f"Skip unused {name}")
            continue
        try:
            state_dict[name].copy_(param)
        except Exception as e:
            raise Exception(f"Error loading {name}: {e}")

    model.load_state_dict(state_dict)

def freeze_encoder_params(model):
    """ Freeze encoder parameters for finetuning """
    for param in model.position_embedding.parameters():
        param.requires_grad = False
    for param in model.transformer.parameters():
        param.requires_grad = False

def finetune(*args, **kwargs) -> str:
    """ Finetune UniSRec model """
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

    # Init config
    config = Config(
        model=UniSRec,
        dataset="project",
        config_file_list=[
            os.path.join(dir_path, "configs", "Sequential.yaml"),
            os.path.join(dir_path, "configs", "SASRec.yaml"),
            os.path.join(dir_path, "configs", "UniSRec.yaml"),
            os.path.join(dir_path, "configs", "finetune.yaml"),
        ],
    )
    config["data_path"] = os.path.join(dir_path, config["data_path"])

    init_seed(config['seed'], config['reproducibility'])
    logger.info(config)

    # Init dataset
    dataset = UniSRecDataset(config)
    train_data, _, _ = data_preparation(config, dataset)

    # Init model
    model = UniSRec(config, train_data._dataset).to(config['device'])
    pretrained_path = os.path.join(dir_path, "saved", "UniSRec-pretrain.pth")
    load_pretrained_weights(model, pretrained_path, config['device'])

    # Freeze encoder parameters
    freeze_encoder_params(model)

    # Init trainer
    all_item_num = len(dataset.item_text)
    trainer = FinetuneTrainer(config, model, all_item_num)

    # Train model
    best_valid_score, best_valid_result = trainer.fit(
        train_data, saved=True, show_progress=config['show_progress'], force_save=True
    )

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('best valid score', 'yellow') + f': {best_valid_score}')

    return trainer.saved_model_file

if __name__ == "__main__":
    result = finetune()
    print(result)
