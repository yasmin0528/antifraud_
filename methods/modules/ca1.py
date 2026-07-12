import torch
import torch.nn as nn


class CA1Encoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, dropout=0.1,
                 encoder_type="gru", pooling="attention"):
        super().__init__()
        if encoder_type != "gru" or pooling != "attention":
            raise ValueError("CA1 v1 supports only encoder_type='gru' and pooling='attention'")
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(hidden_dim, 1)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, sequence, sequence_len, padding_mask=None):
        if padding_mask is None:
            steps = torch.arange(sequence.shape[1], device=sequence.device)
            padding_mask = steps.unsqueeze(0) < (sequence.shape[1] - sequence_len.unsqueeze(1))
        padding_mask = padding_mask.bool()
        encoded, _ = self.gru(sequence)
        attn_logits = self.attention(encoded).squeeze(-1)
        attn_logits = attn_logits.masked_fill(padding_mask, -1e9)
        all_pad = padding_mask.all(dim=1)
        attn = torch.softmax(attn_logits, dim=1).masked_fill(padding_mask, 0.0)
        attn = torch.where(all_pad.unsqueeze(1), torch.zeros_like(attn), attn)
        embedding = torch.sum(encoded * attn.unsqueeze(-1), dim=1)
        embedding = torch.where(all_pad.unsqueeze(1), torch.zeros_like(embedding), embedding)
        embedding = self.dropout(embedding)
        score_micro_logit = self.score_head(embedding)
        return embedding, score_micro_logit, torch.sigmoid(score_micro_logit)
