from .unisrec import UniSRecDataset, PretrainUniSRecDataset, PretrainDataLoader

PRETRAIN_LOADER_MAP = {
    "UniSRec": PretrainDataLoader
}

PRETRAIN_DATASET_MAP = {
    "UniSRec": PretrainUniSRecDataset
}

DATASET_MAP = {
    "UniSRec": UniSRecDataset,
}

__all__ = ["DATASET_MAP", "PRETRAIN_DATASET_MAP"]