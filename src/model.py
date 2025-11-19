import tensorflow as tf
from tensorflow.keras import layers, models

def build_training_model(vocab_size: int, maxlen_questions: int, maxlen_answers: int, embedding_dim=200, lstm_units=200):
    """
    Build the full seq2seq training model (teacher-forcing).
    Returns a compiled Keras Model.
    """
    # Encoder
    encoder_inputs = layers.Input(shape=(maxlen_questions,))
    encoder_embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    _, state_h, state_c = layers.LSTM(lstm_units, return_state=True)(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = layers.Input(shape=(maxlen_answers,))
    decoder_embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = layers.Dense(vocab_size, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

    return model