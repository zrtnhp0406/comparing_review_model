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

        # 🔥 1. Encoder (thiếu trong code bạn)
        self.encoder = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True
        )

        # 🔥 2. Projection (giống ESIM)
        self.projection = nn.Sequential(
            nn.Linear(8 * hidden, hidden),
            nn.ReLU()
        )
        self.fusion_late = GatedFusion(2 * hidden)
        # 🔥 3. Decoder (composition layer)
        self.decoder = nn.LSTM(
            input_size=hidden,  # Input from projection layer
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True
        )

        # 🔥 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden, 2 * hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * hidden, hidden),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels)
        )

    def encode(self, input_ids, mask):
        out = self.bert(input_ids=input_ids, attention_mask=mask)
        return out.last_hidden_state

    def forward(self, input_ids_1, mask_1, input_ids_2, mask_2):

        # ===== 1. BERT =====
        emb1 = self.encode(input_ids_1, mask_1)
        emb2 = self.encode(input_ids_2, mask_2)

        # ===== 2. BiLSTM Encoder =====
        enc1, _ = self.encoder(emb1)
        enc2, _ = self.encoder(emb2)

        # ===== 3. Attention =====
        attention = torch.matmul(enc1, enc2.transpose(1, 2))

        # 🔥 Mask (rất quan trọng)
        mask_2_exp = mask_2.unsqueeze(1)
        mask_1_exp = mask_1.unsqueeze(1)

        attention_1 = attention.masked_fill(mask_2_exp == 0, -1e9)
        attention_2 = attention.transpose(1, 2).masked_fill(mask_1_exp == 0, -1e9)

        weight_1 = torch.softmax(attention_1, dim=-1)
        weight_2 = torch.softmax(attention_2, dim=-1)

        # ===== 4. Alignment =====
        align1 = torch.matmul(weight_1, enc2)
        align2 = torch.matmul(weight_2, enc1)

        # ===== 5. Enhancement =====
        enhance1 = torch.cat([enc1, align1, enc1 - align1, enc1 * align1], dim=-1)
        enhance2 = torch.cat([enc2, align2, enc2 - align2, enc2 * align2], dim=-1)
        # ===== 6. Projection =====
        m1 = self.projection(enhance1)
        m2 = self.projection(enhance2)

        # ===== 7. Decoder =====
        v1, _ = self.decoder(m1)
        v2, _ = self.decoder(m2)

        # ===== 8. Pooling =====
        v1_avg = (v1 * mask_1.unsqueeze(-1)).sum(1) / mask_1.sum(1, keepdim=True)
        v1_max = v1.masked_fill(mask_1.unsqueeze(-1) == 0, -1e9).max(1)[0]

        v2_avg = (v2 * mask_2.unsqueeze(-1)).sum(1) / mask_2.sum(1, keepdim=True)
        v2_max = v2.masked_fill(mask_2.unsqueeze(-1) == 0, -1e9).max(1)[0]

        # ===== 9. Final =====
        v1 = self.fusion_late(v1_avg, v1_max)
        v2 = self.fusion_late(v2_avg, v2_max)
        v = torch.cat([v1, v2], dim=-1)
        logits = self.classifier(v)
        return logits
