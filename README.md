# Neural Network Architectures

This document provides a brief overview of various neural network architectures and their components. Below are descriptions and corresponding images for each architecture, including dimensionality changes for an example input where applicable.

## Summary

1. [Simple MLP](#simple-mlp)
2. [MLP with Batch Normalization](#mlp-with-batch-normalization)
3. [MLP with Batch Normalization and Dropout](#mlp-with-batch-normalization-and-dropout)
4. [CNN frontend and MLP with Batch Normalization and Dropout](#cnn-frontend-and-mlp-with-batch-normalization-and-dropout)
5. [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
6. [XVector-MLP](#xvector-mlp)
7. [CNN-GRU Architecture](#cnn-gru-architecture)
8. [CNN Encoder-Decoder](#cnn-encoder-decoder)
9. [Encoder-MLP](#encoder-mlp)
10. [Frozen Encoder with MLP](#frozen-encoder-with-mlp)
11. [Vanilla RNN](#vanilla-rnn)
12. [RNN with Attention Muli-layer Perceptron (MLP)](#rnn-with-attention-multi-layer-perceptron-mlp)
13. [RNN with Gated Recurrent Unit (GRU)](#rnn-with-gated-recurrent-unit-gru)
14. [Encoder-Decoder with Attention MLP](#encoder-decoder-with-attention-mlp)
15. [Transformer](#transformer)
16. [wev2vec](#wev2vec)
17. [HuBERT](#hubert)
18. [wevLM](#wevLM)
19. [ASR](#asr)
20. [Beamsearch](#beam-search)
21. [TTS](#tts)
22. [NLU](#nlu)
23. [GLM](#glm)


a) [SpeechBrain Workflow](#speechbrain-workflow)


## Simple MLP
![Simple MLP](images/MLP.png)
A basic MLP architecture consisting of linear layers and activation functions. It is used for tasks like digit classification.

| Stage               | Operation               | Input Dimensions            | Output Dimensions           | Explanation                                          |
|---------------------|-------------------------|-----------------------------|-----------------------------|------------------------------------------------------|
| Raw Input           | Raw data input         | (Batch size, Input features)|                             | Input raw features to the MLP                      |
| Linear Layer        | Fully connected layer  | (Batch size, Input features)| (Batch size, Hidden units)  | Applies learned weights and biases                 |
| Output Layer        | Fully connected layer  | (Batch size, Hidden units)  | (Batch size, Classes)       | Maps hidden units to output classes                |

## MLP with Batch Normalization
![MLP with Batch Normalization](images/MLP-BN.png)
A simple MLP architecture that includes batch normalization for faster convergence and better performance.

| Stage               | Operation               | Input Dimensions            | Output Dimensions           | Explanation                                          |
|---------------------|-------------------------|-----------------------------|-----------------------------|------------------------------------------------------|
| Raw Input           | Raw data input         | (Batch size, Input features)|                             | Input raw features to the MLP                      |
| Linear Layer        | Fully connected layer  | (Batch size, Input features)| (Batch size, Hidden units)  | Applies learned weights and biases                 |
| BatchNorm Layer     | Batch normalization    | (Batch size, Hidden units)  | (Batch size, Hidden units)  | Normalizes activations for stability               |

## MLP with Batch Normalization and Dropout
![MLP with Batch Normalization and Dropout](images/MLP-BN-DO.png)
A fully connected architecture with batch normalization and dropout layers for improved training stability and regularization.

| Stage               | Operation               | Input Dimensions            | Output Dimensions           | Explanation                                          |
|---------------------|-------------------------|-----------------------------|-----------------------------|------------------------------------------------------|
| Raw Input           | Raw data input         | (Batch size, Input features)|                             | Input raw features to the MLP                      |
| Linear Layer        | Fully connected layer  | (Batch size, Input features)| (Batch size, Hidden units)  | Applies learned weights and biases                 |
| BatchNorm Layer     | Batch normalization    | (Batch size, Hidden units)  | (Batch size, Hidden units)  | Normalizes activations for stability               |
| Dropout Layer       | Regularization layer   | (Batch size, Hidden units)  | (Batch size, Hidden units)  | Randomly zeroes activations to prevent overfitting |
| Output Layer        | Fully connected layer  | (Batch size, Hidden units)  | (Batch size, Classes)       | Maps hidden units to output classes                |

## CNN frontend and MLP with Batch Normalization and Dropout
![CNN frontend and MLP](images/CNN-MLP-BN-DO.png)
This architecture uses a CNN frontend for feature extraction, followed by an MLP with batch normalization and dropout for classification. The combination helps leverage spatial feature extraction and stable training.

| Stage               | Operation               | Input Dimensions            | Output Dimensions           | Explanation                                          |
|---------------------|-------------------------|-----------------------------|-----------------------------|------------------------------------------------------|
| Raw Input           | Input image data       | (Batch size, Channels, Height, Width)|                             | Input grayscale images                              |
| Conv2D Layer        | Convolutional layer    | (Batch size, Channels, Height, Width)| (Batch size, Filters, Reduced Height, Reduced Width)| Extracts spatial features                          |
| Pooling Layer       | Downsampling layer     | (Batch size, Filters, Reduced Height, Reduced Width)| (Batch size, Filters, Further Reduced Height, Further Reduced Width)| Reduces spatial resolution                        |
| Flatten Layer       | Vectorization          | (Batch size, Filters, Further Reduced Height, Further Reduced Width)| (Batch size, Flattened Features)| Converts feature maps to vectors                  |
| MLP with BatchNorm & Dropout | Fully connected layers | (Batch size, Flattened Features)| (Batch size, Classes)       | Performs final classification                     |

## Convolutional Neural Network (CNN)
![Convolutional Neural Network](images/CNN.png)
A traditional CNN uses convolutional layers followed by pooling layers and activation functions to extract hierarchical features from input images. The features are flattened and passed to fully connected layers for classification.

| Stage               | Operation               | Input Dimensions            | Output Dimensions           | Explanation                                          |
|---------------------|-------------------------|-----------------------------|-----------------------------|------------------------------------------------------|
| Raw Input           | Input image data       | (Batch size, Channels, Height, Width)     |                             | Input grayscale images                              |
| Conv2D Layer        | Convolutional layer    | (Batch size, 1, 28, 28)     | (Batch size, Filters, 26, 26)    | Extracts spatial features                          |
| Pooling Layer       | Downsampling layer     | (Batch size, Filters, 26, 26)    | (Batch size, Filters, 13, 13)    | Reduces spatial resolution                        |
| Flatten Layer       | Vectorization          | (Batch size, Filters, 13, 13)    | (Batch size, 5408)          | Converts feature maps to vectors                  |

## XVector-MLP
![XVector-MLP](images/XVector-MLP.png)
A combination of feature extraction using x-vectors and classification using an MLP. It is typically used in speaker recognition tasks.

| Stage               | Operation               | Input Dimensions            | Output Dimensions           | Explanation                                          |
|---------------------|-------------------------|-----------------------------|-----------------------------|------------------------------------------------------|
| Raw Input           | Input audio features   | (Batch size, Audio features)|                             | Input audio feature vectors                        |
| XVector Extraction  | Feature extraction     | (Batch size, Audio features)| (Batch size, Embedding size)| Extracts compact speaker embeddings               |
| MLP Layer           | Fully connected layer  | (Batch size, Embedding size)| (Batch size, Classes)       | Performs final classification                     |

## CNN-GRU Architecture
![CNN-GRU Architecture](images/CNN-GRU.png)
This architecture combines convolutional layers for feature extraction and a GRU (Gated Recurrent Unit) layer for sequential modeling. It ends with a mean-over-time operation and an MLP for classification.

| Stage               | Operation               | Input Dimensions            | Output Dimensions           | Explanation                                          |
|---------------------|-------------------------|-----------------------------|-----------------------------|------------------------------------------------------|
| Raw Input           | Input sequential data  | (Batch size, 40, Sequence length)|                             | Sequential data input                              |
| Conv1D Layer        | Convolutional layer    | (Batch size, 40, Sequence length)| (Batch size, Filters, Sequence length - 2)| Extracts temporal features                        |
| MaxPool1D Layer     | Downsampling layer     | (Batch size, Filters, Sequence length - 2)| (Batch size, Filters, (Sequence length - 2) / 2)| Reduces temporal resolution                      |
| GRU Layer           | Recurrent layer        | (Batch size, Filters, (Sequence length - 2) / 2)| (Batch size, 128)           | Captures sequential dependencies                  |

## CNN Encoder-Decoder
![CNN Encoder-Decoder](images/ENCODER-DECODER.png)
An encoder-decoder structure with Conv1D layers for feature extraction and transposed convolutions for reconstructing input data. Dropout and Leaky ReLU are used for regularization and activation.

| Stage               | Operation               | Input Dimensions            | Output Dimensions           | Explanation                                          |
|---------------------|-------------------------|-----------------------------|-----------------------------|------------------------------------------------------|
| Raw Input           | Input sequential data  | (Batch size, Channels, Time steps) |                             | Input time-series data                              |
| Encoder Conv1D Layer| Convolutional layer    | (Batch size, 1, Time steps) | (Batch size, 40, Time steps / 160)| Compresses temporal features                       |
| Decoder ConvTranspose1D | Transposed convolution | (Batch size, 40, Time steps / 160) | (Batch size, 1, Time steps) | Reconstructs time-series data                      |

## Encoder-MLP
![Encoder-MLP](images/ENCODER-MLP.png)
This architecture uses an encoder for feature extraction from audio signals, followed by an MLP for classification. It is commonly applied to audio-based tasks.

| Stage               | Operation               | Input Dimensions            | Output Dimensions           | Explanation                                          |
|---------------------|-------------------------|-----------------------------|-----------------------------|------------------------------------------------------|
| Raw Input           | Input audio signals    | (Batch size, 1, Time steps) |                             | Input sequential audio                              |
| Encoder Layer       | Feature extraction     | (Batch size, 1, Time steps) | (Batch size, Features)      | Extracts compact representations                   |
| MLP Layer           | Fully connected layer  | (Batch size, Features)      | (Batch size, Classes)       | Performs final classification                     |

## Frozen Encoder with MLP
![Frozen Encoder with MLP](images/FROZEN-ENCODER.png)
A pre-trained frozen encoder is used to extract features, which are then passed to an MLP for classification. This approach is efficient when working with limited data.

| Stage               | Operation               | Input Dimensions            | Output Dimensions           | Explanation                                          |
|---------------------|-------------------------|-----------------------------|-----------------------------|------------------------------------------------------|
| Raw Input           | Input image data       | (Batch size, Channels, Height, Width)|                             | Input image data                                   |
| Frozen Encoder      | Pre-trained model      | (Batch size, Channels, Height, Width)| (Batch size, Features)      | Extracts compact image features                   |
| MLP Layer           | Fully connected layer  | (Batch size, Features)      | (Batch size, Classes)       | Performs final classification                     |

---

## Vanilla RNN
![Vanilla RNN](images/Vanilla-RNN.png)
A simple RNN architecture with recurrent connections that allow information to persist over time. It is used for sequential data processing tasks.
![Vanilla RNN Dimenstion](images/Vanilla-RNN-Dim.png)

## RNN with Attention Multi-layer Perceptron (MLP)
![RNN Attention MLP](images/RNN-Attention-MLP.png)
An RNN architecture with an attention mechanism that focuses on relevant parts of the input sequence. The attention weights are computed using an MLP.
![RNN Attention MLP Dimenstion](images/RNN-Attention-MLP-Dim.png)
![RNN Attention MLP Summmary](images/RNN-Attention-MLP-Sum.png)

## RNN with Gated Recurrent Unit (GRU)
![RNN GRU](images/RNN-GRU.png)
An RNN architecture with GRU cells that have gating mechanisms to control the flow of information. It helps in capturing long-range dependencies in sequential data.
![RNN GRU Dimenstion](images/RNN-GRU-Sum.png)

## Encoder-Decoder with Attention MLP
![Encoder-Decoder Attention MLP](images/MLP-ATTNTION.png)
An encoder-decoder architecture with an attention mechanism that focuses on relevant parts of the input sequence. The attention weights are computed using an MLP.
![Encoder-Decoder Attention MLP Dimenstion](images/MLP-ATTENTION-sum.png)
![Encoder-Decoder Attention MLP Summmary](images/MLP-ATTENTION-dim.png)

## Transformer
![Transformer](images/Transformer.png)
A transformer architecture that uses self-attention mechanisms to capture long-range dependencies in sequential data. It consists of encoder and decoder layers with multi-head attention.
![Transformer Dimenstion](images/Transformer-dim-1.png)
![Transformer Dimenstion](images/Transformer-dim-2.png)
![Transformer Dimenstion](images/Transformer-dim-3.png)
![Transformer Summary](images/Transformer-sum.png)

## wev2vec
![wev2vec](images/wev2vec.png)
A wev2vec architecture that learns word embeddings by predicting the context of words in a sentence. It uses a skip-gram model with negative sampling for training.
![wev2vec summary](images/wev2vec-sum.png)

## HuBERT
![HuBERT](images/HuBERT1.png)
![HuBERT](images/HuBERT2.png)
A HuBERT architecture that uses a transformer-based model for self-supervised learning. It leverages masked language modeling and contrastive learning to learn representations.
![HuBERT summary](images/HuBERT-sum.png)

## wevLM
![wevLM](images/wevLM.png)
A wevLM architecture that uses a transformer-based model for language modeling. It predicts the next word in a sequence given the context words.

## ASR
![ASR](images/ASR1.png)
An ASR (Automatic Speech Recognition) architecture that uses a combination of convolutional layers and recurrent layers to transcribe spoken language into text. The model processes audio features and generates a sequence of text tokens.
![ASR](images/ASR2.png)
![ASR](images/ASR-sum.png)


## Beam Search
![Beam Search](images/Beam-Search.png)
A beam search architecture that is used in sequence-to-sequence models for generating text. It maintains a fixed number of candidate sequences (beam width) during decoding to find the most likely output sequence.

## TTS
![TTS](images/TTS.png)
A TTS (Text-to-Speech) architecture that converts text input into speech output. It uses a combination of convolutional layers and recurrent layers to generate audio waveforms from text sequences.

## NLU
![nlu](images/NLU1.png)
![nlu](images/NLU2.png)
An NLU (Natural Language Understanding) architecture that uses an encoder-decoder structure. The encoder processes the input text to extract meaningful representations, while the decoder generates outputs such as intent classification, entity recognition, or other language understanding tasks. This architecture is commonly used in conversational AI and NLP applications.
![nlu summary](images/NLU-sum.png)

## GLM
![glm](images/GLM.png)
A GLM (Generalized Language Model) architecture that uses a transformer-based model for language modeling tasks. It is designed to handle various NLP tasks such as text generation, summarization, and question answering.
![glm summary](images/GLM-sum.png)
![glm summary](images/GLM-sum-2.png)




## SpeechBrain Workflow
![SpeechBrain Workflow](images/SpeechBrain-Workflow.png)
The SpeechBrain workflow illustrates the end-to-end process of building and training a speech processing model. It includes data preprocessing, feature extraction, model training, and evaluation stages. The workflow is designed to be modular and flexible, allowing users to customize each component according to their specific needs.


Feel free to edit this README to include more details or customize the descriptions.
