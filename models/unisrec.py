#%%
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections.abc import Iterable
from models.sasrec import SASRec
from transformers import AutoModel, AutoTokenizer
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#%%
class FixedTextEncoder(nn.Module):
    def __init__(self, config):
        super(FixedTextEncoder, self).__init__()
        self.device = config['device']
        self.encoder_model = config['encoder_model']
        self.fixed_item_emb_path = os.path.join(
            os.path.dirname(__file__), 
            os.pardir,
            config['fixed_item_emb_path']
        )
        self.emb_size = config['embedding_size']
        self.fixed_encoder_tokenizer = AutoTokenizer.from_pretrained(self.encoder_model)
        self.fixed_encoder_model = AutoModel.from_pretrained(self.encoder_model, device_map = self.device)
        for param in self.fixed_encoder_model.parameters():
            param.requires_grad = False
        self.load_fixed_item_embs(self.fixed_item_emb_path)

    def load_fixed_item_embs(self, file_path):
        if os.path.exists(file_path):
            logger.warning(f"Loading fixed item embeddings from {file_path}")
            self.fixed_item_embs = torch.load(file_path, map_location = torch.device(self.device))
            self.fixed_item_embs.update({0: torch.zeros((1, self.emb_size)).to(self.device)})
        else:
            self.fixed_item_embs = {0: torch.zeros((1, self.emb_size)).to(self.device)}
        
    def update_fixed_item_embs(self, all_item_map):
        for item_id, item_text in tqdm(all_item_map.items()):
            if item_id not in self.fixed_item_embs:
                if item_text == "" or item_text is None:
                   self.fixed_item_embs[item_id] = torch.zeros((1, self.emb_size)).to(self.device)
                else:
                    self.fixed_item_embs[item_id] = self.forward(item_text)
        torch.save(self.fixed_item_embs, self.fixed_item_emb_path)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        return output
    
    def get_item_emb(self, item_id):
        """ get fixed item embedding """
        if item_id.item() in self.fixed_item_embs:
            item_emb = self.fixed_item_embs[item_id.item()].squeeze(0)
        else:
            return torch.zeros((1, self.emb_size)).to(self.device)
        return item_emb
    
    def get_item_seq_emb(self, item_seq_batch):
        """ calc mean embedding of item_seq """
        embedding_batch = torch.zeros((item_seq_batch.size(0), item_seq_batch.size(1), self.emb_size)).to(self.device)
        for i, item_seq in enumerate(item_seq_batch):  # Batch
            for j, item_id in enumerate(item_seq):  # max_seq_len
                if item_id.item() != 0:
                    embedding_batch[i, j] = self.get_item_emb(item_id)
        return embedding_batch
    
    def get_all_item_embs(self):
        values = []
        max_item_id = max(self.fixed_item_embs.keys())
        for i in range(max_item_id + 1):
            if i not in self.fixed_item_embs:
                self.fixed_item_embs[i] = torch.zeros((1, self.emb_size)).to(self.device)
            values.append(self.fixed_item_embs[i])
        all_item_embs = torch.cat(values, dim=0)
        return all_item_embs

    def forward(self, item_texts):
        if hasattr(self, "fixed_item_embs") and self.fixed_item_embs is not None:
            if isinstance(item_texts, str):
                item_texts = [item_texts]
            if not isinstance(item_texts, Iterable):
                logging.warning(f"item_texts is not iterable: {item_texts}")
                item_texts = [str(item_texts)]
            if all(item_seq in self.fixed_item_embs for item_seq in item_texts):
                return self.fixed_item_embs[item_texts]
        text_encoded = self.fixed_encoder_tokenizer(
            item_texts,
            padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(self.device)
        output = self.fixed_encoder_model(**text_encoded)
        output = self.mean_pooling(output, text_encoded['attention_mask'])
        return output 
        
class PWLayer(nn.Module):
    """ 
    tilde xi = (xi - b)᠂ W1
    """
    def __init__(self, input_dim, output_dim, dropout_prob = 0.2, device = None) -> None:
        super(PWLayer, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.bias = nn.Parameter(torch.zeros(input_dim)).to(device)
        self.weight = nn.Linear(input_dim, output_dim).to(device)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.dropout(x)
        output = self.weight(x - self.bias)
        return output

class MoEAdapter(nn.Module):
    def __init__(self, config) -> None:
        super(MoEAdapter, self).__init__()
        self.device = config['device']
        self.is_noisy_gate = config['MoE_is_noisy_gate']

        input_dim = config['PW_input_dim']
        output_dim = config['PW_output_dim']
        dropout_prob = config['PW_dropout_prob']
        self.n_experts = config['MoE_experts']
        
        self.experts = nn.ModuleList([PWLayer(input_dim, output_dim, dropout_prob, self.device) for _ in range(self.n_experts)])
        self.gate_weight = nn.Parameter(torch.zeros(input_dim, self.n_experts)).to(self.device)
        self.noise_weight = nn.Parameter(torch.zeros(input_dim, self.n_experts)).to(self.device)

    def gating(self, x, noise_epsilon = 1e-2):
        gate_logits = x @ self.gate_weight 
        if self.training and self.is_noisy_gate:
            norm = torch.randn_like(gate_logits).to(self.device)
            noise_logits =  norm * (F.softplus(x @ self.noise_weight) + noise_epsilon)
            gate_logits = gate_logits + noise_logits
        gate_logits = F.softmax(gate_logits, dim=-1)
        return gate_logits

    def forward(self, x):
        gate_logits = self.gating(x) # [Batch, Experts]
        experts_out = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_experts)] # [Batch, 1, v_dim]
        experts_out = torch.cat(experts_out, dim=-2) 
        v_out = (gate_logits.unsqueeze(-1) * experts_out).sum(dim=-2)
        return v_out

class UniSRec(SASRec):
    def __init__(self, config, dataset):
        super(UniSRec, self).__init__(config, dataset)
        self.train_stage = config['stage'] # pretrain or finetune
        if self.train_stage == 'finetune':
            self.parameter_efficient = config['parameter_efficient'] # inductive or transductive 
        self.device = config['device']
        self.pt_lambda = config['pretrain_lambda']
        self.temperature = config['temperature']
        self.ITEM_SEQ_AUG = self.ITEM_SEQ + config['ITEM_SEQ_AUG_SUFFIX']
        self.ITEM_SEQ_AUG_LEN = self.ITEM_SEQ_LEN + config['ITEM_SEQ_AUG_SUFFIX']
        self.ITEM_TEXT_MAP = config['ITEM_TEXT_MAP_FIELD']

        text_map = self.get_all_item_text_map(dataset)

        self.fixed_encoder = FixedTextEncoder(config)
        self.moe_adapter = MoEAdapter(config)
        self.fixed_encoder.update_fixed_item_embs(text_map)
        self.all_test_item_embs = copy.deepcopy(self.fixed_encoder.fixed_item_embs)

    def get_all_item_text_map(self, dataset):
        item_text_map = getattr(dataset, self.ITEM_TEXT_MAP)
        max_item_id = max(item_text_map.keys())
        for i in range(max_item_id + 1):
            if i not in item_text_map:
                item_text_map[i] = ""
        
        return item_text_map
        
    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), 
            dtype=torch.long, 
            device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)
        position_emb = self.position_embedding(position_ids)

        item_emb = self.moe_adapter(self.fixed_encoder.get_item_seq_emb(item_seq))
        input_emb = item_emb + position_emb
        input_emb = self.layer_norm(input_emb)
        input_emb = self.dropout(input_emb)

        attention_mask = self.get_attention_mask(item_seq)
        all_encoder_layer = self.transformer(input_emb, attention_mask)
        last_layer = all_encoder_layer[-1]
        seq_output = self.gather_indexes(last_layer, item_seq_len-1)
        output = seq_output.squeeze(1) #[Batch, Emb]

        return output
    
    def pretrain_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        sequence_output = self.forward(item_seq, item_seq_len)
        sequence_output = F.normalize(sequence_output, p=2, dim=1) # [Batch, Emb]

        pos_item = interaction[self.POS_ITEM_ID]
        same_pos_item_matrix = (pos_item.unsqueeze(1) == pos_item.unsqueeze(0))
        same_pos_item_matrix = torch.logical_xor(
            same_pos_item_matrix, 
            torch.eye(pos_item.shape[0], dtype=torch.bool, device=pos_item.device)
        )

        seq_item_loss = self.seq_item_contrastive_task(sequence_output, same_pos_item_matrix, interaction)
        seq_seq_loss = self.seq_seq_contrastive_task(sequence_output, same_pos_item_matrix, interaction)
        loss = seq_item_loss + self.pt_lambda * seq_seq_loss # # BPR
        return loss

    def finetune_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        sequence_output = self.forward(item_seq, item_seq_len)
        sequence_output = F.normalize(sequence_output, p=2, dim=1) # [Batch, Emb]

        test_item_emb = self.fixed_encoder.get_all_item_embs()
        test_item_output = self.moe_adapter(test_item_emb)

        if self.parameter_efficient == 'transductive':
            test_item_output = test_item_output + self.item_embedding.weight
        test_item_output = F.normalize(test_item_output, p=2, dim=1)

        logits = torch.matmul(sequence_output, test_item_output.transpose(0, 1)) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]

        loss = self.loss_fn(logits, pos_items) # CE

        return loss

    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain_loss(interaction)
        elif self.train_stage == 'finetune':
            return  self.finetune_loss(interaction)
        else:
            raise ValueError(f"Invalid stage {self.train_stage}")

    def seq_item_contrastive_task(self, seq_out, same_item_matrix, interaction):
        pos_item = interaction[self.POS_ITEM_ID]
        pos_item = pos_item.unsqueeze(1)
        pos_item_emb = self.moe_adapter(self.fixed_encoder.get_item_seq_emb(pos_item))
        pos_item_emb = pos_item_emb.squeeze(1)
        pos_item_emb = F.normalize(pos_item_emb, dim=1)

        pos_logits = seq_out * pos_item_emb
        pos_logits = pos_logits.sum(dim = 1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_out, pos_item_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(
            same_item_matrix, 
            torch.tensor([0], dtype=torch.float, device=same_item_matrix.device),
            neg_logits
        )
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits).mean()
        return loss
        
    def seq_seq_contrastive_task(self, seq_out, same_item_matrix, interaction):
        item_seq_aug = interaction[self.ITEM_SEQ_AUG]
        item_seq_aug_len = interaction[self.ITEM_SEQ_AUG_LEN]

        seq_aug_out = self.forward(item_seq_aug, item_seq_aug_len)
        seq_aug_out = F.normalize(seq_aug_out, p=2, dim=1)

        pos_logits = seq_out * seq_aug_out
        pos_logits =  pos_logits.sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_out, seq_aug_out.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(
            same_item_matrix, 
            torch.tensor([0], dtype=torch.float, device=same_item_matrix.device),
            neg_logits
        )
        neg_logits = torch.exp(neg_logits).sum(dim=1)
        
        loss = -torch.log(pos_logits / neg_logits).mean()
        return loss
    
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ].to(self.device)
        item_seq_len = interaction[self.ITEM_SEQ_LEN].to(self.device)
        sequence_output = self.forward(item_seq, item_seq_len)
        sequence_output = F.normalize(sequence_output, p=2, dim=1) # [Batch, Emb]

        test_item_emb = self.fixed_encoder.get_all_item_embs()
        test_item_output = self.moe_adapter(test_item_emb)

        if self.parameter_efficient == 'transductive':
            test_item_output = test_item_output + self.item_embedding.weight
        test_item_output = F.normalize(test_item_output, p=2, dim=1)

        logits = torch.matmul(sequence_output, test_item_output.transpose(0, 1))

        return logits



# %%
