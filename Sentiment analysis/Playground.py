from Dataset import *
from Transformer import *
from Main import *

###--------------------------------------------------------------------------------------------###
# Inserisci la recensione da provare qui!!!
input_sentence = """"aaa"""

# se hai cambiato configurazione in Main.py, devi cambiarla anche qua sotto
###--------------------------------------------------------------------------------------------###



cleaned_input = clean_text(input_sentence)


tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
tokenized_input = tokenizer(cleaned_input)
input_indices = [word_to_id.get(word, UNK) for word in tokenized_input]
input_tensor = torch.tensor(input_indices).unsqueeze(0)


# Configurazione del modello, come in Train.py
config = ModelConfig(
    encoder_vocab_size=vocab_size,
    d_embed=128, # lunghezza vettore embedding per ogni parola
    d_ff=4 * 128, # dimensione livello fully-connected feed-forward
    h=4,         # numero di teste di attenzione
    N_encoder=2,  # numero di encoder
    max_seq_len=max_seq_len, #200
    dropout=0.1   # probabilità di dropout
    )

model = Transformer(config, num_classes)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Effettua la predizione
with torch.no_grad():
    output = model(input_tensor)

# Ottieni la classe predetta
predicted_class = torch.argmax(output, dim=1).item()


if predicted_class == 0:
    print("'{}' is a bad review :(".format(input_sentence))
elif predicted_class == 1:
    print("'{}' is a good review :)".format(input_sentence))
else:
    print("Strange!!!! The review '{}' is predicted as class {}, it should be 0 or 1".format(input_sentence, predicted_class))

