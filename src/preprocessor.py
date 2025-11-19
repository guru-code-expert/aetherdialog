import numpy as np
from tensorflow.keras import preprocessing
from typing import Tuple

class DialogPreprocessor:
    """Handles tokenization, padding and creation of encoder/decoder datasets."""

    def __init__(self):
        self.tokenizer = preprocessing.text.Tokenizer(oov_token="<UNK>")
        self.vocab_size = None
        self.maxlen_questions = None
        self.maxlen_answers = None

    def fit(self, questions: list, answers: list):
        """Fit tokenizer on questions + answers and add special tokens."""
        answers_with_tags = [f"<START> {a} <END>" for a in answers]

        self.tokenizer.fit_on_texts(questions + answers_with_tags)
        self.vocab_size = len(self.tokenizer.word_index) + 1

        # Compute sequence lengths
        tok_questions = self.tokenizer.texts_to_sequences(questions)
        tok_answers = self.tokenizer.texts_to_sequences(answers_with_tags)

        self.maxlen_questions = max(len(x) for x in tok_questions)
        self.maxlen_answers = max(len(x) for x in tok_answers)

    def transform(self, questions: list, answers: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            encoder_input_data, decoder_input_data, decoder_output_data
        """
        answers_with_tags = [f"<START> {a} <END>" for a in answers]

        # Encoder input
        encoder_input_data = self._pad(self.tokenizer.texts_to_sequences(questions), self.maxlen_questions)

        # Decoder input (with <START> <END>)
        decoder_input_data = self._pad(self.tokenizer.texts_to_sequences(answers_with_tags), self.maxlen_answers)

        # Decoder target – shift by one, remove <START>
        decoder_target = [seq[1:] for seq in self.tokenizer.texts_to_sequences(answers_with_tags)]
        decoder_target_padded = self._pad(decoder_target, self.maxlen_answers)
        decoder_output_data = preprocessing.utils.to_categorical(decoder_target_padded, self.vocab_size)

        return (
            np.array(encoder_input_data),
            np.array(decoder_input_data),
            np.array(decoder_output_data),
        )

    def _pad(self, sequences: list, maxlen: int) -> list:
        return preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding="post")

    def preprocess_user_input(self, sentence: str) -> np.ndarray:
        """Preprocess a single user sentence for inference."""
        seq = self.tokenizer.texts_to_sequences([sentence.lower()])
        return preprocessing.sequence.pad_sequences(seq, maxlen=self.maxlen_questions, padding="post")