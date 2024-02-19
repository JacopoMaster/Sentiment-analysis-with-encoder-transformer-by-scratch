import torch
import torchtext
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import collections
import numpy as np
from torch.utils.data import DataLoader
import random

#imposto seed per riproducibilità
torch.manual_seed(42)
random.seed(42)

# Caricamento delle stopwords
stop_words = set(stopwords.words('english'))


# Caricamento dei dati IMDB
from torchtext.datasets import IMDB
train_iter, test_iter = IMDB()
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

# Conversione delle etichette in 0 e 1
y_train = torch.tensor([label-1 for (label, text) in train_iter])
y_test  = torch.tensor([label-1 for (label, text) in test_iter])

# Funzione per pulire il testo
def clean_text(text):
    # Rimozione dei tag HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Conversione in minuscolo
    text = text.lower()
    # Rimozione della punteggiatura
    text = re.sub(r'[\W_]+', ' ', text)
    return text

# Applicazione della pulizia del testo ai dataset
train_iter = ((label, clean_text(text)) for label, text in train_iter)
test_iter = ((label, clean_text(text)) for label, text in test_iter)

# Tokenizzazione dei testi e troncamento del numero di parole in ogni testo a max_seq_len
max_seq_len = 200
x_train_texts = [tokenizer(text)[0:max_seq_len] for (label, text) in train_iter]
x_test_texts  = [tokenizer(text)[0:max_seq_len] for (label, text) in test_iter]

# Costruzione del vocabolario e mappatura delle parole agli interi
counter = collections.Counter()
for text in x_train_texts:
    counter.update(text)

vocab_size = 15000  # Numero di parole uniche nel dizionario
most_common_words = np.array(counter.most_common(vocab_size - 2))  # (-2) per i token di padding e unknown
vocab = most_common_words[:,0]

# Indici per i token di padding e unknown
PAD = 0
UNK = 1
word_to_id = {vocab[i]: i + 2 for i in range(len(vocab))}

# Mappatura delle parole nei testi di addestramento e test agli interi
x_train = [torch.tensor([word_to_id.get(word, UNK) for word in text]) for text in x_train_texts]
x_test  = [torch.tensor([word_to_id.get(word, UNK) for word in text]) for text in x_test_texts]

# Padding dei testi in modo che abbiano la stessa lunghezza di sequenza
x_test = torch.nn.utils.rnn.pad_sequence(x_test, batch_first=True, padding_value=PAD)

# Costruzione del dataset compatibile con torch.utils.data.Dataloader
class IMDBDataset:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.features[item], self.labels[item]
    
train_dataset = IMDBDataset(x_train, y_train)
test_dataset  = IMDBDataset(x_test, y_test)   

# Funzione collate_fn da utilizzare in torch.utils.data.DataLoader
# Aggiunge il padding ai testi in ogni batch in modo che abbiano la stessa lunghezza di sequenza
def pad_sequence(batch):
    texts  = [text for text, label in batch]
    labels = torch.tensor([label for text, label in batch])
    texts_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=PAD)
    return texts_padded, labels

# DataLoader per i dati di addestramento e test
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=pad_sequence)


# Calcolo la dimensione per i dataset di validation e test
total_size = len(test_dataset)
validation_size = total_size // 2
test_size = total_size - validation_size

# Genero indici casuali per il dataset
indices = list(range(total_size))
random.shuffle(indices)

# Assegno gli indici per i dataset di validation e test
validation_indices = indices[:validation_size]
test_indices = indices[validation_size:]

# Creo i dataset di validation e test utilizzando gli indici
validation_dataset = [test_dataset[i] for i in validation_indices]
test_dataset = [test_dataset[i] for i in test_indices]

# Creo i DataLoader per il dataset di validation e test
validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=True, collate_fn=pad_sequence)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, collate_fn=pad_sequence)
