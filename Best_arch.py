import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim * 2, dim)

    def forward(self, u, v):
        # u, v: (batch, dim)
        x = torch.cat([u, v], dim=-1)
        g = torch.sigmoid(self.linear(x))
        h = g * u + (1 - g) * v
        return h
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        # x: (batch, seq_len, hidden)
        
        e = self.score(x).squeeze(-1)  # (batch, seq_len)

        # mask padding
        e = e.masked_fill(mask == 0, -1e9)

        alpha = torch.softmax(e, dim=1)  # attention weights

        # weighted sum
        v = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)

        return v
class ESIM_BERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased", freeze_layers=10, num_labels=3):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)
        # Freeze
        if freeze_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(freeze_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
        
        hidden = self.bert.config.hidden_size

        # 🔥 2. Projection (giống ESIM)
        self.projection = nn.Sequential(
            nn.Linear(4 * hidden,2 * hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    
        self.attn = nn.MultiheadAttention(self.bert.config.hidden_size, num_heads=8, batch_first=True)
        self.pooling = AttentionPooling(hidden)
        # 🔥 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden,hidden),
            nn.Tanh(),
            nn.Linear(hidden, num_labels)
        )

    def encode(self, input_ids, mask):
        out = self.bert(input_ids=input_ids, attention_mask=mask)
        return out.last_hidden_state

    def forward(self, input_ids_1, mask_1, input_ids_2, mask_2):

        # ===== 1. BERT =====
        enc1 = self.encode(input_ids_1, mask_1)
        enc2 = self.encode(input_ids_2, mask_2)

        # ===== 4. Alignment =====
        align1, _ = self.attn(enc1, enc2, enc2, key_padding_mask=~mask_2.bool())
        align2, _ = self.attn(enc2, enc1, enc1, key_padding_mask=~mask_1.bool())

        # ===== 5. Enhancement =====
        enhance1 = torch.cat([enc1, align1, enc1 - align1, enc1 * align1], dim=-1)
        enhance2 = torch.cat([enc2, align2, enc2 - align2, enc2 * align2], dim=-1)
        # ===== 6. Projection =====
        m1 = self.projection(enhance1)
        m2 = self.projection(enhance2)
        
        # ===== 8. Pooling =====
        v1 = self.pooling(m1, mask_1)
        v2 = self.pooling(m2, mask_2)
        v = torch.cat([v1, v2], dim=-1)
        logits = self.classifier(v)
        return logits
