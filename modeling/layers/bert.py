import tensorflow as tf

from modeling.config import BertConfig
from modeling.layers.embedding import BertEmbedding
from modeling.layers.encoder_block import EncoderBlock
from modeling.utils import get_shape_list


class Bert(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig):
        super(Bert, self).__init__()
        self.bert_embedding = BertEmbedding(config)
        self.encoder_blocks = [EncoderBlock(config) for _ in range(config.num_hidden_layers)]

    def call(self, input_ids, input_mask=None, token_type_ids=None, training=False):
        batch_size, sequence_length = get_shape_list(input_ids)

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, sequence_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, sequence_length], dtype=tf.int32)

        embedding_output = self.bert_embedding(input_ids, token_type_ids, training)  # (B, S, E)

        attention_mask = tf.cast(tf.reshape(input_mask, [batch_size, 1, sequence_length]), dtype=tf.float32)
        broadcast_ones = tf.ones([batch_size, sequence_length, 1], dtype=tf.float32)
        attention_mask *= broadcast_ones  # (B, S_q, S_k)

        x = embedding_output
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, attention_mask, batch_size, training)
        return x
