# Analisi del Sentiment delle Recensioni di Film

## Overview
Il progetto si concentra sull'analisi del sentiment delle recensioni di film utilizzando il dataset IMDB. L'obiettivo è sviluppare un modello basato sull'architettura Transformer per classificare le recensioni in positive o negative.

## Dataset
Il dataset IMDB contiene 50.000 recensioni, equamente suddivise tra training e test set. Ogni recensione è etichettata come positiva o negativa.

## Preparazione del Dataset
Il dataset viene pulito per rimuovere i tag HTML, convertire il testo in minuscolo e rimuovere la punteggiatura. Successivamente, viene tokenizzato e convertito in sequenze di interi.

## Architettura del Modello
Il modello utilizza solo la parte dell'encoder dell'architettura Transformer, con i seguenti componenti principali:
- Input Embedding
- Positional Encoding
- Encoder
- Linear layer
- softmax

## Addestramento e Valutazione
L'addestramento del modello incorpora un meccanismo di early stopping per prevenire l'overfitting. La valutazione si basa su loss e accuracy, sia durante l'addestramento sia sul test set.

## Risultati
I risultati mostrano che il modello raggiunge un'accuratezza soddisfacente sul test set, con una leggera preferenza per una configurazione del modello più leggera in termini di parametri ma con prestazioni quasi uguali alle configurazioni più complesse.

## Utilizzo
Forniamo uno script per testare il modello su recensioni personali. È necessario utilizzare il modello allenato 'model.pth' con la configurazione specificata.

## Riferimenti
- [Learning word vectors for sentiment analysis, Maas et al., 2011](#)
- [Attention is all you need, Vaswani et al., 2017](#)


