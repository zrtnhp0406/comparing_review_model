import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

class model_Bert(nn.Module):
    def __init__(self, model_name="bert-base-uncased", freeze_layers=10, num_labels=3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        
        if freeze_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(freeze_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False

        hidden = self.bert.config.hidden_size

        # 2. Lớp Projection (Sơ chế sau khi tăng cường thông tin)
        self.projection = nn.Sequential(
            nn.Linear(4 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 3. Lớp Classifier (Phân loại cuối cùng)
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden, hidden),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels)
        )

    def encode(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def forward(self, input_ids_1, mask_1, input_ids_2, mask_2):
        vec_1 = self.encode(input_ids_1, mask_1)
        vec_2 = self.encode(input_ids_2, mask_2)

        # e_ij
        attention_matrix = torch.matmul(vec_1, vec_2.transpose(1, 2))

        weight_1 = torch.softmax(attention_matrix, dim=-1)
        weight_2 = torch.softmax(attention_matrix.transpose(1, 2), dim=-1)

        vec_1_aligned = torch.matmul(weight_1, vec_2)
        vec_2_aligned = torch.matmul(weight_2, vec_1)

        vec_1_combined = torch.cat([
            vec_1, 
            vec_1_aligned, 
            vec_1 - vec_1_aligned, 
            vec_1 * vec_1_aligned
        ], dim=-1)

        vec_2_combined = torch.cat([
            vec_2, 
            vec_2_aligned, 
            vec_2 - vec_2_aligned, 
            vec_2 * vec_2_aligned
        ], dim=-1)
        
        projected_1 = F.relu(self.projection(vec_1_combined))
        projected_2 = F.relu(self.projection(vec_2_combined))

        v1_avg = projected_1.mean(dim=1)
        v1_max = projected_1.max(dim=1)[0]

        v2_avg = projected_2.mean(dim=1)
        v2_max = projected_2.max(dim=1)[0]

        v_final = torch.cat([v1_avg, v1_max, v2_avg, v2_max], dim=-1)

        logits = self.classifier(v_final)
        return logits