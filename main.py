# Created by James Astell (17668733) as a partial fulfilment of the requirements for the
# Degree of BSc(Hons) Computer Science
# Section of code derived from: https://www.tensorflow.org/text/tutorials/transformer

import collections
import logging
import os
import io
import pathlib
import re
import string
import sys
import time
from downloading_data import init_logging
import tqdm as tqdm
import timeit
from pick import pick
import os.path
from os import path
from jiwer import wer
import concurrent.futures
import requests

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

# Initialise logger
init_logging()
logger = logging.getLogger(__name__)

# Access the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


# ******************************************* MODEL FUNCTIONS *******************************************

def tokenize_pairs(pt, en):
    pt = tokenizers.pt.tokenize(pt)
    # Convert from ragged to dense, padding with zeros.
    pt = pt.to_tensor()

    en = tokenizers.en.tokenize(en)
    # Convert from ragged to dense, padding with zeros.
    en = en.to_tensor()
    return pt, en


def make_batches(ds):
    return (
        ds
            .cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))


# val_batches = make_batches(val_examples)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.tokenizer = Encoder(num_layers, d_model, num_heads, dff,
                                 input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.tokenizer(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))


def train_model(EPOCHS):
    training_start_time = timeit.default_timer()  # Start training timer

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> input, tar -> response
        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(
                    f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    with open('status.txt', 'w') as filetowrite:
        filetowrite.write(str(status))

    # Stop training timer
    training_elapsed = timeit.default_timer() - training_start_time
    print("Time taken training:", round(training_elapsed), "sec")


# ******************************************* MODEL TESTING *******************************************

# Main evaluate method
def evaluate(sentence, max_length=40, timeout=10):
    # Add start + end token
    sentence = tf.convert_to_tensor([sentence])
    sentence = tokenizers.pt.tokenize(sentence).to_tensor()

    encoder_input = sentence

    # The first word to the transformer should be the start token
    start, end = tokenizers.en.tokenize([''])[0]
    output = tf.convert_to_tensor([start])
    output = tf.expand_dims(output, 0)

    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # Select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.argmax(predictions, axis=-1)

        output = tf.concat([output, predicted_id], axis=-1)

        # Return the result if the predicted_id is equal to the end token
        if predicted_id == end:
            break

    text = tokenizers.en.detokenize(output)[0]  # shape: ()

    tokens = tokenizers.en.lookup(output)[0]

    return text, tokens, attention_weights


# Generate response using evaluate function
def response(sentence):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        futures.append(executor.submit(evaluate, sentence=sentence, timeout=0.00001))
        for future in concurrent.futures.as_completed(futures):
            try:
                translated_text, translated_tokens, attention_weights = future.result()
            except requests.ConnectTimeout:
                print("ConnectTimeout.")

    # print('Input: %s' % (sentence))
    # print('Predicted response: {}'.format(translated_text))

    return translated_text


# Calculates the word error rate using jiwer
def WER(gt_path, hypothesis_path):
    GTsentences = io.open(gt_path, encoding='UTF-8').read().strip().split('\n')

    Hsentences = io.open(hypothesis_path, encoding='UTF-8').read().strip().split('\n')

    error = wer(GTsentences, Hsentences)
    logger.info("Calculating the word error rate...")
    print("The word error rate is: ", round(error, 2))

    return error


# Main evaluation function
def evaluate_model(filename):
    testing_start_time = timeit.default_timer()  # Start testing timer

    # Open the testing dialogue and split at new line
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]  # Remove \n characters

    # Pass each sentence to response function and add output to new empty list
    empty_list = []
    new_string = ''
    for x in tqdm.tqdm(content, desc='Generating responses based on testing data:'):
        result = response(x)
        new_string = ('{}'.format(result.numpy().decode('utf-8')))
        # new_string = new_string.replace('<end>', '')
        empty_list.append(new_string)

    # File Saving
    file_path = "processed_data/BLEU/machine_translated_dialogue.txt"
    logger.info(f"Saving predicted response data to {file_path}")

    with open(file_path, mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(empty_list))

    logger.info("Saving Complete!")

    gt_path = current_dir + "/processed_data/BLEU/human_translated_dialogue.txt"
    hypothesis_path = current_dir + "/processed_data/BLEU/machine_translated_dialogue.txt"

    # Calculating the word error rate
    WER(gt_path, hypothesis_path)

    # Stop testing timer
    testing_elapsed = timeit.default_timer() - testing_start_time
    print("Time taken testing:", round(testing_elapsed), "sec")


# ******************************************* CANDIDATE MODEL TESTING *******************************************

# Main evaluate method
def candidate_evaluate(sentence, max_length=40, candidate=None, id=None):
    # Add start + end token
    sentence = tf.convert_to_tensor([sentence])
    sentence = tokenizers.pt.tokenize(sentence).to_tensor()

    candidate = tf.convert_to_tensor([candidate])
    candidate = tokenizers.pt.tokenize(candidate)

    encoder_input = sentence

    value = 0
    text = ''

    # The first word to the transformer should be the start token
    start, end = tokenizers.en.tokenize([''])[0]
    output = tf.convert_to_tensor([start])
    output = tf.expand_dims(output, 0)

    candidate_words = []
    if candidate != None:
        test = candidate.to_list()
        for item in test:
            candidate_words += item

    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # Select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        if candidate == None:
            predicted_id_out = tf.argmax(predictions, axis=-1)  # Need raw word input for rest of code to work
        else:
            predicted_id = candidate_words[i]  # For candidate

        predicted_id_out = tf.argmax(predictions, axis=-1)  # Need raw word input for rest of code to work

        pred_shape = predictions[0]  # Changing the shape of the prediction array to allow for manipulation

        value += pred_shape[0][predicted_id].numpy()

        output = tf.concat([output, predicted_id_out], axis=-1)
        text = tokenizers.en.detokenize(output)[0]  # shape: ()

        # Return the result if the predicted_id is equal to the end token
        if predicted_id == 3:
            return text, value / i, id

    text = tokenizers.en.detokenize(output)[0]  # shape: ()
    # tokens = tokenizers.en.lookup(output)[0]

    return text, value / i, id


# Read and pre-process candidate specific data as different format
def candidate_load_dataset(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    lines = text.strip().split('\n')

    allCandidates = []  # Will contain all 10 candidate responses
    candidates = []  # temp
    contexts = []  # The input sentence for response to be generated

    for i in range(0, len(lines)):
        if lines[i].startswith("CONTEXT:"):
            candidate = lines[i][8:]
            contexts.append(candidate)
            continue

        elif len(lines[i].strip()) == 0:
            if i > 0: allCandidates.append(candidates)
            candidates = []

        else:
            candidate = lines[i][12:]
            candidates.append(candidate)

    allCandidates.append(candidates)
    return allCandidates, contexts


# Calculating rank value for performance metrics
def rank_value(target_value, unsorted_distribution):
    sorted_distribution = sorted(unsorted_distribution, reverse=True)  # Sort the distribution list
    for i in range(0, len(sorted_distribution)):
        value = sorted_distribution[i]  # Value equal to candidate distance value
        if value == target_value:
            return 1 / (i + 1)  # Calculate distance away from ground truth
    return None


# Main evaluation function
def candidate_evaluate_model(filename_testdata):
    testing_start_time = timeit.default_timer()  # Start testing timer

    # Init variables/list
    candidates, contexts = candidate_load_dataset(filename_testdata)
    correct_predictions = 0
    total_predictions = 0
    cumulative_mrr = 0
    recall_at_1 = None
    mrr = None
    ref_empty_list = []
    resp_empty_list = []

    # For all candidates using tqdm progress bar
    for i in tqdm.tqdm(range(0, len(contexts)), desc='Evaluating model'):
        total_predictions += 1
        target_value = 0
        context = contexts[i]
        reference = candidates[i][0]
        ref_empty_list.append(reference)
        distribution = []
        jobs = []

        # Using 'concurrent.features. to enable parallel and reduce execution time
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     jobs.append(executor.submit(candidate_evaluate, context, 40, None, 0))
        #     for j in range(0, len(candidates[i])):
        #         jobs.append(executor.submit(candidate_evaluate, context, 40, candidates[i][j], (j + 1)))
        #
        # for future in concurrent.futures.as_completed(jobs):
        #     candidate_sentence, value_candidate, id = future.result()  # Return function variables

        for j in range(0, len(candidates[i])):
            candidate_sentence, value_candidate, id = candidate_evaluate(context, 40, candidates[i][j], (j))
            # First id is the response
            if id == 0:
                response = ('{}'.format(candidate_sentence.numpy().decode('utf-8')))
                resp_empty_list.append(response)  # Add to response list for BLEU evaluation
            # Add the value to the distribution before being passed into the rank_value function
            else:
                distribution.append(value_candidate)

            if id == 1:
                target_value = value_candidate  # Ground-truth response

        rank = rank_value(target_value, distribution)
        cumulative_mrr += rank  # Running total of rank to be divided later
        correct_predictions += 1 if rank == 1 else 0  # Running total of correct predictions

        recall_at_1 = correct_predictions / total_predictions  # Final recall@1 calc
        mrr = cumulative_mrr / total_predictions  # Final mrr calc

    # File Saving
    ref_file_path = "processed_data/BLEU/human_translated_dialogue.txt"
    resp_file_path = "processed_data/BLEU/machine_translated_dialogue.txt"

    logger.info(f"Saving reference dialogue data to {ref_file_path}")
    with open(ref_file_path, mode='wt', encoding='utf-8') as ref_myfile:
        ref_myfile.write('\n'.join(ref_empty_list))

    logger.info(f"Saving response dialogue data to {resp_file_path}")
    with open(resp_file_path, mode='wt', encoding='utf-8') as resp_myfile:
        resp_myfile.write('\n'.join(resp_empty_list))

    logger.info("Saving Complete!")

    # Print results
    print("The Recall@1 value is: " + str(recall_at_1))
    print("The Mean Reciprocal Rank value is: " + str(mrr))

    # Calculating the word error rate + print in function
    WER(ref_file_path, resp_file_path)

    # Stop testing timer
    testing_elapsed = timeit.default_timer() - testing_start_time
    print("Time taken testing:", round(testing_elapsed), "sec")


# ******************************************* USER INPUT *******************************************

def action():
    all_data = ""
    path_to_file = ""
    status = ""
    epoch_option = ""

    # Gather status of model
    model_status = '\nCurrent status: '
    with open('status.txt', 'r') as file:
        model_status += file.read()

    # Allow the user to choose the domain being trained + report status
    model_question = 'What action would you like to perform? (Evaluation must be performed according to previously ' \
                     'trained model)\n' + model_status
    model_answers = ['Train on pre-processed data', 'Train on provided candidate data',
                     'Measure the performance of pre-processed data model using validation dataset',
                     'Measure the performance of candidate data model using validation dataset',
                     'Evaluate pre-processed data model', 'Evaluate candidate data model']
    model_option, index = pick(model_answers, model_question)

    # Loop for training on pre-processed data
    if index == 0:
        epoch_question = 'How many epochs to train? '
        epoch_answer = [5, 10, 25, 50, 100]
        epoch_option, index_epoch = pick(epoch_answer, epoch_question)

        status = "The currently saved model is based on the 'pre-processed dataset' over " + str(
            epoch_option) + " epochs."

        if path.exists(checkpoint_path):
            logger.info("Removing previous checkpoints...\n")
            for filename in os.listdir(checkpoint_path):
                os.remove(checkpoint_path + "/" + filename)

        path_to_file = 'processed_data/train/all_training_dialogue.csv'

    # Loop for training on candidate data
    if index == 1:
        epoch_question = 'How many epochs to train? '
        epoch_answer = [5, 10, 25, 50, 100]
        epoch_option, index_epoch = pick(epoch_answer, epoch_question)

        status = "The currently saved model is based on the 'candidate dataset' over " + str(epoch_option) + " epochs."

        if path.exists(checkpoint_path):
            logger.info("Removing previous checkpoints...\n")
            for filename in os.listdir(checkpoint_path):
                os.remove(checkpoint_path + "/" + filename)

        path_to_file = 'processed_data/candidate/dstc8-train.csv'

    if index == 4:
        path_to_file = 'processed_data/train/all_training_dialogue.csv'

    if index == 5:
        path_to_file = 'processed_data/candidate/dstc8-train.csv'

    return all_data, path_to_file, status, epoch_option, index


# ******************************************* MODEL PARAMETERS *******************************************

checkpoint_path = "./checkpoints/train"

all_data, path_to_file, status, epoch_option, index = action()

train_examples = tf.data.experimental.CsvDataset(path_to_file, ["", ""])

for pt_examples, en_examples in train_examples.batch(3).take(1):
    for pt in pt_examples.numpy():
        print(pt.decode('utf-8'))

    print()

    for en in en_examples.numpy():
        print(en.decode('utf-8'))

path_to_dir = 'tokenizer_model'

tokenizers = tf.saved_model.load(path_to_dir)

print([item for item in dir(tokenizers.en) if not item.startswith('_')])

for en in en_examples.numpy():
    print(en.decode('utf-8'))

encoded = tokenizers.en.tokenize(en_examples)

for row in encoded.to_list():
    print(row)

round_trip = tokenizers.en.detokenize(encoded)
for line in round_trip.numpy():
    print(line.decode('utf-8'))

tokens = tokenizers.en.lookup(encoded)
print(tokens)

BUFFER_SIZE = 20000
BATCH_SIZE = 64

train_batches = make_batches(train_examples)

n, d = 2048, 512
pos_encoding = positional_encoding(n, d)
print(pos_encoding.shape)
pos_encoding = pos_encoding[0]

# Juggle the dimensions for the plot
pos_encoding = tf.reshape(pos_encoding, (n, d // 2, 2))
pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
pos_encoding = tf.reshape(pos_encoding, (d, n))

x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
create_padding_mask(x)

x = tf.random.uniform((1, 3))
temp = create_look_ahead_mask(x.shape[1])
print(temp)

np.set_printoptions(suppress=True)

temp_k = tf.constant([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10],
                      [0, 0, 10]], dtype=tf.float32)  # (4, 3)

temp_v = tf.constant([[1, 0],
                      [10, 0],
                      [100, 5],
                      [1000, 6]], dtype=tf.float32)  # (4, 2)

# This `query` aligns with the second `key`,
# so the second `value` is returned.
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

# This query aligns with a repeated key (third and fourth),
# so all associated values get averaged.
temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

# This query aligns equally with the first and second key,
# so their values get averaged.
temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

temp_q = tf.constant([[0, 0, 10],
                      [0, 10, 0],
                      [10, 10, 0]], dtype=tf.float32)  # (3, 3)
print_out(temp_q, temp_k, temp_v)

temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
out, attn = temp_mha(y, k=y, q=y, mask=None)
print(out.shape, attn.shape)

sample_ffn = point_wise_feed_forward_network(512, 2048)
print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)

sample_encoder_layer = EncoderLayer(512, 8, 2048)

sample_encoder_layer_output = sample_encoder_layer(
    tf.random.uniform((64, 43, 512)), False, None)

print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)

sample_decoder_layer = DecoderLayer(512, 8, 2048)

sample_decoder_layer_output, _, _ = sample_decoder_layer(
    tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
    False, None, None)

print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)

sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                         dff=2048, input_vocab_size=8500,
                         maximum_position_encoding=10000)
temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                         dff=2048, target_vocab_size=8000,
                         maximum_position_encoding=5000)
temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

output, attn = sample_decoder(temp_input,
                              enc_output=sample_encoder_output,
                              training=False,
                              look_ahead_mask=None,
                              padding_mask=None)

print(output.shape, attn['decoder_layer2_block2'].shape)

sample_transformer = Transformer(
    num_layers=2, d_model=512, num_heads=8, dff=2048,
    input_vocab_size=8500, target_vocab_size=8000,
    pe_input=10000, pe_target=6000)

temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                               enc_padding_mask=None,
                               look_ahead_mask=None,
                               dec_padding_mask=None)

print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

temp_learning_rate_schedule = CustomSchedule(d_model)

plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size(),
    target_vocab_size=tokenizers.en.get_vocab_size(),
    pe_input=1000,
    pe_target=1000,
    rate=dropout_rate)

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# ******************************************* MAIN *******************************************

if (index == 0) or (index == 1):
    train_model(epoch_option)

# if index == 2:
#     path_to_data_val = "processed_data/val/input_val_dialogue.txt"
#     # Restore the latest checkpoint.
#     if ckpt_manager.latest_checkpoint:
#         ckpt.restore(ckpt_manager.latest_checkpoint)
#         print('Latest checkpoint restored!!')
#     evaluate_model(path_to_data_val)
#
# if index == 3:
#     path_to_data_val = current_dir + "/processed_data/candidate/dstc8-val-candidates.txt"
#     # Restore the latest checkpoint.
#     if ckpt_manager.latest_checkpoint:
#         ckpt.restore(ckpt_manager.latest_checkpoint)
#         print('Latest checkpoint restored!!')
#     candidate_evaluate_model(path_to_data_val)

if index == 4:
    path_to_data_test = 'processed_data/test/input_testing_dialogue.txt'
    # Restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    evaluate_model(path_to_data_test)

if index == 5:
    path_to_data_test = 'processed_data/candidate/dstc8-test-candidates.txt'
    # Restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    candidate_evaluate_model(path_to_data_test)
