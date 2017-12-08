#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import seq2seq

in_wordcount = 500
out_wordcount = 200
in_seq_len = 20
out_seq_len = 20

epochs = 200

if __name__ == '__main__':
    with open("data.json") as f:
        json_data = json.load(f)
        question_text = [row['question'] for row in json_data]
        answer_text = [row['answer'] for row in json_data]
        dims = seq2seq.TrainingDims(in_wordcount, out_wordcount, in_seq_len, out_seq_len)
        data = seq2seq.preprocess(question_text, answer_text, dims)
        encoder, decoder = seq2seq.train_model(data, dims, optimizer_name='rmsprop', learning_rate=0.001, epochs=epochs, latent_dim=128)