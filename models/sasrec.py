#%%
import torch
import torch.nn as nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss

class SASRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)
        self.device = torch.device(config['device'])
        self.hidden_size = config['hidden_size']
        self.layer_norm_eps = config['layer_norm_eps']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.n_layers = config['num_blocks']
        self.n_heads = config['num_heads'] 
        self.inner_size = config['feed_forward_size']
        self.hidden_act = config['hidden_act']
        self.loss_type = config['loss_type']
        self.init_normal_std = config['init_normal_std']

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size, device=self.device)
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, device=self.device)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps, device=self.device)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.transformer = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.hidden_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        ).to(self.device)
        if self.loss_type == "BPR":
            self.bpr_gamma = config['bpr_gamma']
            self.loss_fn = BPRLoss(gamma = self.bpr_gamma)
        elif self.loss_type == "CE":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        self.apply(self._init_weights)
        

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.init_normal_std)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), 
            dtype=torch.long, 
            device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)

        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.layer_norm(input_emb)
        input_emb = self.dropout(input_emb)
        
        attention_mask = self.get_attention_mask(item_seq) # left-to-right uni-directional 
        all_encoder_layers = self.transformer(input_emb, attention_mask)
        last_layer = all_encoder_layers[-1]

        # left-to-right uni-directional, therefore the last item has potential to represent the sequence
        output = self.gather_indexes(last_layer, item_seq_len-1)
        output = output.squeeze(1) # [Batch, Emb]

        return output
    
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        sequence_output = self.forward(item_seq, item_seq_len)

        pos_item = interaction[self.POS_ITEM_ID] # true label

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_item_emb = self.item_embedding(pos_item)
            neg_item_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(sequence_output * pos_item_emb, dim = -1) # [Batch]
            neg_score = torch.sum(sequence_output * neg_item_emb, dim = -1) # [Batch]
            loss = self.loss_fn(pos_score, neg_score)
            return loss
        elif self.loss_type == 'CE':
            all_item_emb = self.item_embedding.weight # [item_num, Emb]
            logits = torch.matmul(sequence_output, all_item_emb.transpose(0, 1)) # [Batch, Emb][Emb, item_num] = [Batch, item_num]
            loss = self.loss_fn(logits, pos_item)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        sequence_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        item_rec_score = torch.sum(sequence_output * test_item_emb, dim = -1)
        return item_rec_score
    
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        sequence_output = self.forward(item_seq, item_seq_len)
        all_item_emb = self.item_embedding.weight
        all_item_rec_scores = torch.matmul(sequence_output, all_item_emb.transpose(0, 1))
        return all_item_rec_scores
        
