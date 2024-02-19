import math
import torch
import torch.nn as nn
from Module import *

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  # Normalizzazione per evitare che i valori siano troppo grandi

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = Dropout(dropout)

        pe = torch.zeros(max_len, d_model)   
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Calcolo dei valori della codifica posizionale, come nel paper
        pe[:, 0::2] = torch.sin(position * div_term)  # Seno per posizioni pari
        pe[:, 1::2] = torch.cos(position * div_term)  # Coseno per posizioni dispari
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # Aggiunta della codifica posizionale, non aggiorno in backpropagation
        return self.dropout(x) 

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__() 
        self.linear1 = Linear(d_model, d_ff)  # Primo strato lineare
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_ff, d_model)  # Secondo strato lineare

    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))  # Applicazione della funzione di attivazione ReLU
        x = self.linear2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_embed, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        assert d_embed % h == 0, "Cambia configurazione :( \n La dimensione dell'embedding ({}) deve essere divisibile per il numero di teste di attenzione (h = {}).".format(d_embed, h)
        self.d_k = d_embed // h
        self.d_embed = d_embed
        self.h = h
        self.WQ = Linear(d_embed, d_embed)
        self.WK = Linear(d_embed, d_embed)
        self.WV = Linear(d_embed, d_embed)
        self.linear = Linear(d_embed, d_embed)
        self.dropout = Dropout(dropout)

    def forward(self, x_query, x_key, x_value, mask=None):
        nbatch = x_query.size(0)  # Dimensione del batch
        # Proiezioni lineari per ottenere query, chiave e valore multi-head
        query = self.WQ(x_query).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)
        key = self.WK(x_key).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)
        value = self.WV(x_value).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)
        # Calcolo dell'attenzione
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Applicazione della maschera se presente
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        p_atten = torch.nn.functional.softmax(scores, dim=-1)
        p_atten = self.dropout(p_atten)
        x = torch.matmul(p_atten, value)
        x = x.transpose(1, 2).contiguous().view(nbatch, -1, self.d_embed)
        x = self.linear(x)
        return x




class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_embed = config.d_embed
        self.input_embedding = InputEmbedding(config.encoder_vocab_size, self.d_embed)
        self.positional_encoding = PositionalEncoding(self.d_embed, config.max_seq_len, config.dropout)
        # Creazione di una lista di EncoderBlock passando i parametri necessari
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model=config.d_embed, d_ff=config.d_ff, h=config.h, dropout=config.dropout) 
            for _ in range(config.N_encoder)
        ])
        self.norm = LayerNormalization(config.d_embed)

    def forward(self, input, mask=None):
        x = self.input_embedding(input)
        x = self.positional_encoding(x)
        for block in self.encoder_blocks:
            x = block(x, mask)
        x = self.norm(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, h: int, dropout: float):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(h, d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x, mask=None):
        # Prima sub-layer: Multi-Head Attention + Add & Norm
        attn_output = self.attention(x, x, x, mask)  # Applico l'attenzione
        x = x + self.dropout1(attn_output)  # Applico Dropout al risultato dell'attenzione e fai l'Add
        x = self.norm1(x)  # Normalizzazione Layer

        # Seconda sub-layer: Feed Forward + Add & Norm
        ff_output = self.feed_forward(x)  # Applico il Feed Forward Network
        x = x + self.dropout2(ff_output)  # Applico Dropout al risultato del Feed Forward e fai l'Add
        x = self.norm2(x)  # Normalizzazione Layer
        return x

class Transformer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.encoder = Encoder(config)
        self.linear = Linear(config.d_embed, num_classes)

    def forward(self, x, pad_mask=None):
        x = self.encoder(x, pad_mask)
        return  self.linear(torch.mean(x, -2))  

    