import tensorflow as tf

from modeling.config import BertConfig
from modeling.utils import create_initializer, get_shape_list


class MaskedLanguageModel(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, word_embedding):
        super(MaskedLanguageModel, self).__init__()
        self.word_embedding = word_embedding
        self.transform = tf.keras.layers.Dense(units=config.hidden_size,
                                               kernel_initializer=create_initializer(config.initializer_range))
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.output_bias = self.add_weight(name='output_bias', shape=[config.vocab_size],
                                           initializer=tf.zeros_initializer())

    def call(self, context, mask_lm_positions, training=False):
        # B: Batch size, P: Max predictions per sequence, H: Hidden size, V: Vocab size
        context = self._gather_mlm_context(context, mask_lm_positions)  # (B, S, H) -> (B * P, H)
        # TODO: Is gathering the best idea? How about transform the whole sequence then ignoring useless sequence?
        context = self.layer_norm(self.transform(context))
        logits = tf.matmul(context, self.word_embedding, transpose_b=True) + self.output_bias  # (B * P, V)

        return logits

    def _gather_mlm_context(self, context, mask_lm_positions):
        batch_size, sequence_length, hidden_size = get_shape_list(context)
        offset = tf.reshape(tf.range(batch_size, dtype=tf.int32) * sequence_length, [-1, 1]) + mask_lm_positions

        flat_offset = tf.reshape(offset, [-1])
        flat_context = tf.reshape(context, [-1, hidden_size])
        return tf.gather(flat_context, flat_offset)


class NextSentencePrediction(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig):
        super(NextSentencePrediction, self).__init__()
        self.pooler = tf.keras.layers.Dense(units=config.hidden_size,
                                            kernel_initializer=create_initializer(config.initializer_range))
        self.pooler_activation = tf.keras.layers.Activation('tanh')
        self.output_weight = self.add_weight(name='output_weight', shape=[config.hidden_size, 2],
                                             initializer=create_initializer(config.initializer_range))
        self.output_bias = self.add_weight(name='output_bias', shape=[2],
                                           initializer=create_initializer(config.initializer_range))

    def call(self, context, training=False):
        pooled_output = self.pooler_activation(self.pooler(context[:, 0]))
        logits = tf.matmul(pooled_output, self.output_weight) + self.output_bias

        return logits
