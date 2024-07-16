import torch
from torch import nn
from transformers import RobertaModel, GPT2Model
from utils import eos_pooling

class CBL(nn.Module):
    def __init__(self, concept_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(768, concept_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(concept_dim, concept_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return x

class RobertaCBL(nn.Module):
    def __init__(self, concept_dim, dropout):
        super().__init__()
        self.preLM = RobertaModel.from_pretrained('roberta-base')
        for p in self.preLM.parameters():
            p.requires_grad = True
        self.projection = nn.Linear(768, concept_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(concept_dim, concept_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t):
        text_features = self.preLM(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).last_hidden_state[:, 0, :]
        projected = self.projection(text_features)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return x

class GPT2CBL(nn.Module):
    def __init__(self, concept_dim, dropout):
        super().__init__()
        self.preLM = GPT2Model.from_pretrained('gpt2')
        for p in self.preLM.parameters():
            p.requires_grad = True
        self.projection = nn.Linear(768, concept_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(concept_dim, concept_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t):
        text_features = self.preLM(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).last_hidden_state
        text_features = eos_pooling(text_features, t["attention_mask"])
        projected = self.projection(text_features)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return x

class Roberta_Baseline(nn.Module):
    def __init__(self, class_num, projection_dim, dropout):
        super().__init__()
        self.preLM = RobertaModel.from_pretrained('roberta-base')
        for p in self.preLM.parameters():
            p.requires_grad = True
        self.projection = nn.Linear(768, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, class_num)

    def forward(self, t):
        text_features = self.preLM(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).last_hidden_state[:, 0, :]
        projected = self.projection(text_features)
        x = self.gelu(projected)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class GPT2_Baseline(nn.Module):
    def __init__(self, class_num, projection_dim, dropout):
        super().__init__()
        self.preLM = GPT2Model.from_pretrained('gpt2')
        for p in self.preLM.parameters():
            p.requires_grad = True
        self.projection = nn.Linear(768, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, class_num)

    def forward(self, t):
        text_features = self.preLM(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).last_hidden_state
        text_features = eos_pooling(text_features, t["attention_mask"])
        projected = self.projection(text_features)
        x = self.gelu(projected)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class MLP(nn.Module):
    def __init__(self, class_num, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(768, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(projection_dim, projection_dim)
        self.fc2 = nn.Linear(projection_dim, class_num)

    def forward(self, t):
        projected = self.projection(t)
        x = self.gelu(projected)
        x = self.fc1(x)
        x = self.dropout(x)
        x = x + projected
        x = self.fc2(x)
        return x