import os
import logging
import torch
from recbole.config import Config
from recbole.utils import init_seed, init_logger
from recbole.trainer.trainer import PretrainTrainer
from datasets.samplers.seq_sampler import BasicSeqSampler
from models.unisrec import UniSRec
from datasets.unisrec import PretrainDataLoader, PretrainUniSRecDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_prior_model(model, checkpoint_path, device):
    """ Load prior model weights into UniSRec model """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = model.state_dict()

    for name, param in checkpoint['state_dict'].items():
        if name.startswith('item_embedding'):
            logging.info(f"Skip unused {name}")
            continue
        try:
            state_dict[name].copy_(param)
        except Exception as e:
            logging.error(f"Error loading {name}: {e}")
            raise Exception(e)

    model.load_state_dict(state_dict)

def init_dataloader(config, dataset):
    """ Initialize the dataloader for pretraining """
    train_dataset = dataset.build()[0]
    sampler = BasicSeqSampler(dataset=train_dataset, distribution="uniform", alpha=1.0)
    return PretrainDataLoader(config, train_dataset, sampler, False)

def pretrain(*args, **kwargs) -> str:
    """ Pretrain UniSRec model """
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

    # Init config
    config = Config(
        model=UniSRec,
        dataset="project",
        config_file_list=[
            os.path.join(dir_path, "configs", "Sequential.yaml"),
            os.path.join(dir_path, "configs", "SASRec.yaml"),
            os.path.join(dir_path, "configs", "UniSRec.yaml"),
            os.path.join(dir_path, "configs", "pretrain.yaml"),
        ],
    )
    config["data_path"] = os.path.join(dir_path, config["data_path"])

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger.info(config)

    # Init dataset and model
    dataset = PretrainUniSRecDataset(config)
    model = UniSRec(config, dataset).to(config['device'])

    # Load prior model weights
    prior_model_path = os.path.join(dir_path, "checkpoints", "UniSRec-prior.pth")
    load_prior_model(model, prior_model_path, config['device'])

    # Init data loader
    training_data = init_dataloader(config, dataset)

    # Start training
    trainer = PretrainTrainer(config, model)
    trainer.pretrain(training_data, show_progress=True)

    return model

if __name__ == "__main__":
    model = pretrain()
