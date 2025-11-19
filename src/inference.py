import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from .preprocessor import DialogPreprocessor

class Chatbot:
    """
    Wrapper class that loads a trained model and provides a simple .respond() method.
    """
    def __init__(self, model_path: str = None, preprocessor: DialogPreprocessor = None):
        if model_path is None:
            model_path = "models/aetherdialog_lstm.pkl"

        self.preprocessor = preprocessor
        self.enc_model, self.dec_model = self._load_or_build_inference_models(model_path)

    def _load_or_build_inference_models(self, training_model_path: str):
        # Load the full training model first
        full_model = tf.keras.models.load_model(training_model_path)

        # Extract dimensions
        embedding_dim = full_model.layers[2].output_shape[-1]  # Embedding dim
        lstm_units = full_model.layers[3].units

        vocab_size = full_model.output_shape[-1]
        maxlen_answers = full_model.input[1].shape[1]

        # Re-build encoder inference model
        encoder_inputs = full_model.input[0]
        encoder_embedding_layer = full_model.layers[2]
        encoder_lstm_layer = full_model.layers[3]
        encoder_embedding = encoder_embedding_layer(encoder_inputs)
        _, state_h, state_c = encoder_lstm_layer(encoder_embedding)
        encoder_model = models.Model(encoder_inputs, [state_h, state_c])

        # Re-build decoder inference model
        decoder_inputs = full_model.input[1]
        decoder_state_input_h = layers.Input(shape=(lstm_units,))
        decoder_state_input_c = layers.Input(shape=(lstm_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_embedding = full_model.layers[4](decoder_inputs)  # same embedding layer
        decoder_lstm = full_model.layers[5]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_embedding, initial_state=decoder_states_inputs
        )
        decoder_dense = full_model.layers[6]
        decoder_outputs = decoder_dense(decoder_outputs)

        decoder_model = models.Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs, state_h, state_c]
        )

        return encoder_model, decoder_model

    def respond(self, user_input: str, max_length: int = 50) -> str:
        if self.preprocessor is None:
            raise ValueError("Preprocessor not set.")

        input_seq = self.preprocessor.preprocess_user_input(user_input)
        states = self.enc_model.predict(input_seq, verbose=0)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.preprocessor.tokenizer.word_index.get("start", 1)

        stop_condition = False
        decoded_sentence = ""

        while not stop_condition:
            output_tokens, h, c = self.dec_model.predict([target_seq] + states, verbose=0)

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.preprocessor.tokenizer.index_word.get(sampled_token_index, "")

            if sampled_word in {"end", ""} or len(decoded_sentence.split()) >= max_length:
                stop_condition = True
            else:
                decoded_sentence += " " + sampled_word

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states = [h, c]

        return decoded_sentence.strip()