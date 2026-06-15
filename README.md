# Sentiment Analysis with Simple Recurrent Neural Network (RNN)

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Example Prediction](#example-prediction)

## Introduction
This project demonstrates the implementation of a simple Recurrent Neural Network (RNN) for sentiment analysis of movie reviews. The model is trained on the IMDB movie review dataset, which contains text reviews labeled as either positive or negative sentiment.

## Project Overview
The primary goal of this project is to classify movie reviews into positive or negative sentiments using a neural network. A SimpleRNN architecture is employed, leveraging word embeddings to capture semantic relationships within the text data. The model is built using TensorFlow/Keras.

## Dataset
- **Name**: IMDB Movie Review Dataset
- **Description**: Contains 50,000 highly polarized movie reviews for binary sentiment classification. 25,000 reviews are used for training and 25,000 for testing.
- **Characteristics**: Each review is preprocessed into a sequence of integers, where each integer represents a specific word in a vocabulary. The vocabulary size is limited to `max_features` (10,000 most frequent words).

## Preprocessing
1.  **Loading Data**: The `imdb.load_data()` function is used to load the dataset, restricting the vocabulary to the top `max_features` words.
2.  **Word Index Mapping**: A `word_index` dictionary is obtained to map words to their integer representations, and a `reverse_word_index` is created for decoding.
3.  **Padding Sequences**: Reviews, being of variable length, are padded to a fixed maximum length (`max_len = 500`) using `sequence.pad_sequences()`. This ensures uniform input size for the RNN.

### Custom Preprocessing Functions
-   `decode_review(encoded_review)`: Converts a sequence of word indices back into a human-readable review string.
-   `preprocess_text(text)`: Takes a raw text review, converts it to lowercase, tokenizes it into words, converts words to their integer IDs, and then pads the sequence to `max_len`.

## Model Architecture
The sentiment analysis model is a `Sequential` Keras model with the following layers:
1.  **Embedding Layer**: Maps integer-encoded words into dense vectors of fixed size (128 dimensions). `input_length` is set to `max_len`.
    ```python
    model.add(Embedding(max_features, 128, input_length=max_len))
    ```
2.  **SimpleRNN Layer**: A recurrent layer with 128 units and `relu` activation function. This layer processes the sequence of embedded words.
    ```python
    model.add(SimpleRNN(128, activation='relu'))
    ```
3.  **Dense Layer**: A single output neuron with `sigmoid` activation for binary classification (positive/negative sentiment).
    ```python
    model.add(Dense(1, activation='sigmoid'))
    ```

### Model Summary
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 500, 128)          1280000   
 rnn (SimpleRNN)             (None, 128)               32896     
 dense (Dense)               (None, 1)                 129       
=================================================================
Total params: 1313025 (5.01 MB)
Trainable params: 1313025 (5.01 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

## Training
-   **Optimizer**: `adam`
-   **Loss Function**: `binary_crossentropy` (suitable for binary classification)
-   **Metrics**: `accuracy`
-   **Epochs**: 10 (with early stopping)
-   **Batch Size**: 32
-   **Validation Split**: 20% of training data used for validation.
-   **Early Stopping**: An `EarlyStopping` callback monitors `val_loss` with `patience=5` and `restore_best_weights=True` to prevent overfitting.

## Dependencies
-   `numpy`
-   `tensorflow` (including `keras.datasets`, `keras.preprocessing.sequence`, `keras.models`, `keras.layers`, `keras.callbacks`)

## Usage
To use the trained model for predicting the sentiment of a new movie review:

1.  **Load the Model**: If the model is saved, load it using `tensorflow.keras.models.load_model('imdb_rnn.h5')`.
2.  **Define `preprocess_text` and `predict_sentiment` functions**: Ensure the `decode_review`, `preprocess_text` (with `word_index`, `reverse_word_index`, `sequence`, `maxlen`), and `predict_sentiment` functions are defined in your environment.

    ```python
    def decode_review(encoded_review):
      return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

    def preprocess_text(text):
      words = text.lower().split()
      encoded_review = [word_index.get(word, 2) + 3 for word in words]
      padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
      return padded_review

    def predict_sentiment(review):
      preprocessed_input = preprocess_text(review)
      prediction = model.predict(preprocessed_input)
      sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
      return sentiment, prediction[0][0]
    ```

3.  **Make a Prediction**: Call `predict_sentiment()` with your review string.

## Example Prediction
```python
example_review = "This movie was fantastic! The acting was great and the plot was thrilling"
sentiment, score = predict_sentiment(example_review)

print(f'Review: {example_review}')
print(f'Sentiment: {sentiment}')
print(f'Prediction Score: {score}')
```

**Output:**
```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step
Review: This movie was fantastic! The acting was great and the plot was thrilling
Sentiment: Positive
Prediction Score: 0.6661213636398315
```

This output indicates that the model predicted a positive sentiment for the given review with a prediction score of approximately 0.666.
