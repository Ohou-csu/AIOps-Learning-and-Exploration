# -*- coding: utf-8 -*-
# @Time : 2020/9/9 6:14 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : model.py
# @Software: PyCharm
from abc import ABC

import tensorflow as tf
from tensorflow_addons.text.crf import crf_log_likelihood
from transformers import TFBertModel


class NerModel(tf.keras.Model, ABC):
    def __init__(self, configs, vocab_size, num_classes):
        super(NerModel, self).__init__()
        self.use_bert = configs.use_bert
        self.finetune = configs.finetune
        if self.use_bert and self.finetune:
            self.bert_model = TFBertModel.from_pretrained('bert-base-chinese')
        self.use_bigru = configs.use_bigru
        self.embedding = tf.keras.layers.Embedding(vocab_size, configs.embedding_dim, mask_zero=True)
        self.hidden_dim = configs.hidden_dim
        self.dropout_rate = configs.dropout
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.bigru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.hidden_dim, return_sequences=True))
        self.dense = tf.keras.layers.Dense(num_classes)
        self.transition_params = tf.Variable(tf.random.uniform(shape=(num_classes, num_classes)))

    @tf.function
    def call(self, inputs, inputs_length, targets, training=None):
        if self.use_bert:
            if self.finetune:
                embedding_inputs = self.bert_model(inputs[0], attention_mask=inputs[1])[0]
            else:
                embedding_inputs = inputs
        else:
            embedding_inputs = self.embedding(inputs)

        outputs = self.dropout(embedding_inputs, training)

        if self.use_bigru:
            outputs = self.bigru(outputs)

        logits = self.dense(outputs)
        tensor_targets = tf.convert_to_tensor(targets, dtype=tf.int32)
        log_likelihood, self.transition_params = crf_log_likelihood(
            logits, tensor_targets, inputs_length, transition_params=self.transition_params)
        return logits, log_likelihood, self.transition_params
