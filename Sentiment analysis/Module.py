import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        # Inizializza i parametri del layer lineare
        self.in_features = in_features
        self.out_features = out_features
        # Pesi del layer (tensore dei pesi)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            # Bias del layer (tensore dei bias)
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # Resetta i parametri del layer
        self.reset_parameters()

    def reset_parameters(self):
        # Inizializzazione dei pesi usando Kaiming
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # mitiga il problema del vanishing o exploding gradients
        if self.bias is not None:
            # Inizializzazione dei bias secondo una distribuzione uniforme
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Applicazione della trasformazione lineare
        return torch.matmul(input, self.weight.t()) + self.bias if self.bias is not None else torch.matmul(input, self.weight.t())



class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding, self).__init__()
        # Inizializza l'embedding
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Pesi dell'embedding (tensore degli embedding)
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, input):
        # Applicazione dell'embedding
        return torch.embedding(self.weight, input)

class Dropout(nn.Module):
    def __init__(self, p):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, input):
        # Applicazione del dropout
        return torch.dropout(input, self.p, self.training)



