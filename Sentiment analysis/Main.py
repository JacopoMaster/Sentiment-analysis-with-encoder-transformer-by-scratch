from Transformer import *
from Dataset import *
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import torchinfo
import matplotlib.pyplot as plt

# imposto seed per riproducibilità
torch.manual_seed(42)


# Verifica la disponibilità della GPU e imposta il dispositivo di esecuzione
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Numero di classi nel problema di classificazione
num_classes = 2

# con dataclass non devo definire il costruttore, basta indicare i campi
@dataclass
class ModelConfig:
    encoder_vocab_size: int
    d_embed: int
    # d_ff è la dimensione del livello fully-connected feed-forward
    d_ff: int
    # h è il numero di teste di attenzione
    h: int
    N_encoder: int
    max_seq_len: int
    dropout: float


# per creare modello 
def make_model(config):
    model = Transformer(config, num_classes).to(DEVICE)
    return model

# Funzione per addestrare il modello per un'epoca
def train_epoch(model, dataloader, epoch):
    model.train()
    losses, acc, count = [], 0, 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, (x, y) in pbar:
        optimizer.zero_grad()
        features = x.to(DEVICE)
        labels = y.to(DEVICE)
        # Creazione di una maschera di padding per gestire le sequenze di lunghezza variabile
        pad_mask = (features == PAD).view(features.size(0), 1, 1, features.size(-1))
        pred = model(features, pad_mask)

        loss = loss_fn(pred, labels).to(DEVICE)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        acc += (pred.argmax(1) == labels).sum().item()
        count += len(labels)
        # aggiorno la barra di progresso ogni 20 batch
        if idx > 0 and idx % 20 == 0:
            pbar.set_description(f'epoch: {epoch}, loss di training:{loss.item():.4f}, acc. di training:{acc/count:.4f}')
    return np.mean(losses), acc / count

# Funzione per addestrare il modello su più epoche con early stopping
def train(model, train_loader, validation_loader, epochs, patience=3):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for ep in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, ep + 1)
        val_loss, val_acc = evaluate(model, validation_loader)
        print(f'Epoch {ep + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Qui puoi anche salvare il modello se vuoi mantenere il miglior modello
            torch.save(model.state_dict(), 'model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Stopping early at epoch {ep + 1}!")
                break
    # Carica il miglior modello prima della valutazione finale
    model.load_state_dict(torch.load('model.pth'))

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    return model

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
   

# Funzione per valutare le prestazioni del modello su un set di dati di validazione
def evaluate(model, dataloader):
    model.eval()
    total_loss, total_acc, total_count = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Test"):
            features = x.to(DEVICE)
            labels = y.to(DEVICE)
            pad_mask = (features == PAD).view(features.size(0), 1, 1, features.size(-1))
            pred = model(features, pad_mask)
            loss = loss_fn(pred, labels).to(DEVICE)
            
            total_loss += loss.item() * labels.size(0)  # Aggiornamento del loss totale
            total_acc += (pred.argmax(1) == labels).sum().item()  # Aggiornamento dell'accuratezza totale
            total_count += labels.size(0)  # Aggiornamento del conteggio totale

    # Calcolo delle medie
    average_loss = total_loss / total_count
    average_accuracy = total_acc / total_count
    return average_loss, average_accuracy



if __name__ == "__main__":

    # Configurazione del modello
    config = ModelConfig(
    encoder_vocab_size=vocab_size,
    d_embed=128, # lunghezza vettore embedding per ogni parola
    d_ff=4 * 128, # dimensione livello fully-connected feed-forward
    h=4,         # numero di teste di attenzione
    N_encoder=2,  # numero di encoder
    max_seq_len=max_seq_len, #200
    dropout=0.1   # probabilità di dropout
    )

    # Creazione del modello
    model = make_model(config)

    # Riepilogo del modello
    print(torchinfo.summary(model))

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Addestramento del modello
    train(model, train_loader, validation_loader, epochs=15)
    
    # Valutazione sul test set, usa questo pezzo solo dopo aver scelto la configurazione finale 
    test_loss, test_acc = evaluate(model, test_loader)
    print(f'Test set: Loss:{test_loss:.4f}, Accuracy:{test_acc:.4f}')







