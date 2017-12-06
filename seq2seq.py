#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for training sequence-to-sequence models in Keras,
based on https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
The main difference is that it uses a word-level model with an Embedding layer,
instead of a character-level model.
"""
import keras
import keras.layers as KL
import keras.models as KM
import keras.preprocessing.sequence as KSeq
import keras.regularizers as KR
import keras.utils as KUtils
import numpy as np
import random
import keras.backend as KB
from typing import List, Tuple, Dict

UNKNOWN = "UNKNOWN_TOK"
START = "START_SEQUENCE"
END = "END_SEQUENCE"
PAD = "PADDING_TOK"

class TrainingDims:
    def __init__(self, max_input_words: int=None, max_ouput_words: int=None, 
                input_sequence_length: int=None, output_sequence_length: int=None) -> None:
        #Number of tokens is number of words plus 4 special tokens: 
        #"START_TOK", "END_TOK", "UNKNOWN_TOK", "PADDING_TOK"

        self.input_tokens = max_input_words + 3
        self.output_tokens = max_ouput_words + 3
        self.input_seq_len = input_sequence_length
        self.output_seq_len = output_sequence_length

class TrainingData:
    def __init__(self, input_data: np.ndarray, output_data: np.ndarray, output_target_data: np.ndarray, 
                input_dict: List[str], output_dict: List[str]) -> None:
        self.input_data = input_data
        self.output_data = output_data
        self.output_target_data = output_target_data
        self.input_dict = input_dict
        self.output_dict = output_dict

    def validate(self, dims: TrainingDims, m: int):
        """Make sure all our data is the right dimensions"""
        input_shape = self.input_data.shape
        expected_input_shape = (m, dims.input_seq_len)
        assert(input_shape == expected_input_shape)

        output_shape = self.output_data.shape
        expected_output_shape = (m, dims.output_seq_len)
        assert(output_shape == expected_output_shape)

        output_target_shape = self.output_target_data.shape
        expected_output_target_shape = (m, dims.output_seq_len, dims.output_tokens)
        assert(output_target_shape == expected_output_target_shape)


def get_index(word: str, wordlist: List[str]) -> int:
    try:
        return wordlist.index(word)
    except ValueError:
        return wordlist.index(UNKNOWN)


def int_encode(texts: List[str], wordlist: List[str], max_sequence_length: int) -> np.ndarray:
    sequence_list = []

    for i, text in enumerate(texts):
        words = text.split()
        words.insert(0, START)
        words.append(END)
        if max_sequence_length:
            words = words[:max_sequence_length+2] #don't bother encoding after max_post_length
        sequence_list.append([get_index(w, wordlist) for w in words])

    # convert wordlist into padded numpy array
    word_array = KSeq.pad_sequences(sequence_list, maxlen=max_sequence_length, 
                            padding='pre', truncating='post')

    return word_array


def create_wordlist(texts: List[str], token_limit: int) -> List[str]:
    word_freq: Dict[str, int] = {}

    for text in texts:
        for word in text.split():
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    wordlist = sorted(word_freq, key=word_freq.get, reverse=True)
    if token_limit:
        wordlist = wordlist[0:token_limit - 4] #leave room for special tokens
    #word at index 0 is padding (b/c pad_sequences pads w/ value 0)
    wordlist.insert(0, PAD)
    wordlist.append(UNKNOWN)
    wordlist.append(START)
    wordlist.append(END)

    return wordlist


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    """copied from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py"""

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def preprocess(in_sequences, out_sequences, dims: TrainingDims) -> TrainingData:
    """
    Preprocess training data into form we can feed to the model.
    out_sequences[i] should be target for in_sequences[i]

    Args:
        in_sequences: List of input sequences
        out_sequences: Corresponding list of output sequences
        dims: Number of tokens, max sequence length
    Returns:
        A TrainingData instance
    """

    in_wordlist = create_wordlist(in_sequences, dims.input_tokens)
    if not(dims.input_tokens):
        dims.input_tokens = len(in_wordlist)

    in_encoded = int_encode(in_sequences, in_wordlist, dims.input_seq_len)
    if not(dims.input_seq_len):
        dims.input_seq_len = in_encoded.shape[1]


    out_wordlist = create_wordlist(out_sequences, dims.output_tokens)
    if not(dims.output_tokens):
        dims.output_tokens = len(out_wordlist)

    #+1 b/c we truncate later
    if dims.output_seq_len:
        output_len = dims.output_seq_len + 1
    else:
        output_len = None
    out_encoded = int_encode(out_sequences, out_wordlist, output_len)

    if not(dims.output_seq_len):
        dims.output_seq_len = out_encoded.shape[1] - 1 

    out_target = KUtils.to_categorical(out_encoded, num_classes=dims.output_tokens)

    #target is shifted over one from answer_array
    out_encoded = out_encoded[:, :-1]
    out_target = out_target[:, 1:]

    data = TrainingData(in_encoded, out_encoded, out_target,
                        in_wordlist, out_wordlist)
    data.validate(dims, len(in_sequences))
    return data


def fit_model(model: KM.Model, data: TrainingData, batch_size: int, epochs: int) -> None:        
    model.fit([data.input_data, data.output_data], data.output_target_data,
                batch_size=batch_size, epochs=epochs, validation_split=0.2)


def inference_model(encoder_inputs: KL.Input, decoder_inputs: KL.Input, encoder_states, 
                        decoder_embed, decoder_lstm, decoder_dense, latent_dim: int=64,) -> Tuple[KM.Model, KM.Model]:
    encoder_model = KM.Model(inputs=encoder_inputs, outputs=encoder_states)

    sampling_state_input_h = KL.Input(shape=(latent_dim,))
    sampling_state_input_c = KL.Input(shape=(latent_dim,))    
    sampling_state_inputs = [sampling_state_input_h, sampling_state_input_c]
    embedded_sampling_inputs = decoder_embed(decoder_inputs)
    sampling_outputs, sampling_state_h, sampling_state_c = decoder_lstm(
        embedded_sampling_inputs, initial_state=sampling_state_inputs)
    sampling_state_outputs = [sampling_state_h, sampling_state_c]
    sampling_sequence_outputs = decoder_dense(sampling_outputs)

    decoder_model = KM.Model([decoder_inputs] + sampling_state_inputs,
                        [sampling_sequence_outputs] + sampling_state_outputs)
    return (encoder_model, decoder_model)    


def train_model(data: TrainingData, dims: TrainingDims, optimizer_name='adam',
                learning_rate=0.001, batch_size: int=64, latent_dim: int=64, dropout=0.0,
                epochs: int=1, print_samples=True, sample_text=None) -> Tuple[KM.Model, KM.Model]:

    # define our model for training
    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(lr=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(lr=learning_rate)
    else:
        raise RuntimeError("nope")

    # define encoder (given a question, generate hidden state)
    encoder_inputs = KL.Input(shape=(None,))
    encoder_embedded = KL.Embedding(dims.input_tokens + 1, latent_dim, mask_zero=True) (encoder_inputs)
    encoder_lstm, state_h, state_c = KL.LSTM(latent_dim,
                                        return_state=True) (encoder_embedded)
    encoder_states = [state_h, state_c]

    # define decoder nodes
    decoder_inputs = KL.Input(shape=(None,))
    decoder_embed = KL.Embedding(dims.output_tokens + 1, latent_dim, mask_zero=True)
    decoder_lstm = KL.LSTM(latent_dim,
        return_sequences=True, return_state=True)
    decoder_dropout = KL.Dropout(dropout)
    decoder_dense = KL.Dense(dims.output_tokens, activation='softmax')

    # define how input flows through decoder nodes to get output
    embedded_inputs = decoder_embed(decoder_inputs)
    decoder_sequence_outputs, _, _ = decoder_lstm(embedded_inputs, initial_state=encoder_states)
    decoder_outputs = decoder_dense(decoder_dropout(decoder_sequence_outputs))

    # training model takes question AND answer inputs and gives answer outputs
    # we define question & answer inputs (not just questions) b/c we use teacher forcing
    model = KM.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

    # compile & train
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')


    if print_samples:
        epochs_remaining = epochs
        while epochs_remaining > 10:
            fit_model(model, data, batch_size=batch_size, epochs=10)
            (encoder, decoder) = inference_model(encoder_inputs, decoder_inputs, encoder_states, 
                        decoder_embed, decoder_lstm, decoder_dense, latent_dim)

            if sample_text:
                sample_in_text = sample_text
            else:
                random_idx = random.randrange(len(data.input_data))
                sample_input = data.input_data[random_idx]
                sample_in_text = " ".join([data.input_dict[i] for i in sample_input])
            #sample_in_text = "This is some test text."
            print("INPUT")
            print("------------")
            print(sample_in_text)
            print("ANSWER")
            print("------------")
            for temp in [0.2, 0.5, 1.0, 1.2]:
                print(sample_sequence(sample_in_text, data, dims, encoder, decoder, 30, temp))
            epochs_remaining -= 10

        fit_model(model, data, batch_size=batch_size, epochs=epochs_remaining)
        (encoder, decoder) = inference_model(encoder_inputs, decoder_inputs, encoder_states, 
                        decoder_embed, decoder_lstm, decoder_dense, latent_dim)

    else:
        fit_model(model, data, batch_size=batch_size, epochs=epochs)
        return inference_model(encoder_inputs, decoder_inputs, encoder_states, 
                        decoder_embed, decoder_lstm, decoder_dense, latent_dim)

    generative_models = (encoder, decoder)
    return generative_models

def sample_sequence(in_text: str, data: TrainingData, dims: TrainingDims,
                    encoder: KM.Model, decoder: KM.Model, 
                    max_length: int, temperature: float=0.2) -> str:

    unknown_idx = get_index(UNKNOWN, data.output_dict)

    #perform input processing on quesion
    in_array = int_encode([in_text], data.input_dict, dims.input_seq_len)

    # Encode question to state vector
    states_value = encoder.predict(in_array)
    
    # initialize answer sequence w/ SEQUENCE_START character
    out_array = np.zeros((1, 1))
    start_index = get_index(START, data.output_dict)
    out_array[0] = start_index
    # now build up target sequence
    sentence = ""
    wordcount = 0
    while True:

        # predict next word
        word_probabilities, h, c = decoder.predict(
            [out_array] + states_value)

        #never choose UNKNOWN token
        word_probabilities[0][0][unknown_idx] = 0
        word_idx = sample(word_probabilities[0][0], temperature)
        try:
            word = data.output_dict[word_idx]
        except:
            word = UNKNOWN
        sentence += word
        sentence += " "
        wordcount += 1

        # break if we've hit stop character or end of sequence
        if word == END or wordcount > max_length:
            break

        # update target sequence
        out_array = np.zeros((1, 1))
        out_array[0] = word_idx

        # update states
        states_value = [h, c]

    return sentence    
