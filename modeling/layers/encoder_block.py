import tensorflow as tf

from modeling.config import BertConfig
from modeling.layers.activation import gelu
from modeling.utils import create_initializer


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig):
        super(EncoderBlock, self).__init__()
        self.self_attention_block = Residual(internal_layer=SelfAttention(config), config=config)
        self.feed_forward_block = Residual(internal_layer=PositionwiseFeedForward(config), config=config)

    def call(self, inputs, attention_mask, batch_size, training=False):
        attention_output = self.self_attention_block(inputs=inputs,
                                                     training=training,
                                                     internal_layer_kwargs=dict(attention_mask=attention_mask,
                                                                                batch_size=batch_size,
                                                                                training=training))
        output = self.feed_forward_block(inputs=attention_output, training=training)
        return output


class Residual(tf.keras.layers.Layer):
    def __init__(self, internal_layer, config: BertConfig):
        super(Residual, self).__init__()
        self.internal_layer = internal_layer
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)

    def call(self, inputs, training=False, internal_layer_kwargs=None):
        if internal_layer_kwargs is None:
            internal_layer_kwargs = dict()
        internal_output = self.internal_layer(inputs, **internal_layer_kwargs)
        return self.layer_norm(self.dropout(internal_output, training) + inputs)


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.size_per_head = int(config.hidden_size / self.num_attention_heads)

        if config.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f'The hidden size {config.hidden_size} is not a multiple '
                f'of the number of attention heads {self.num_attention_heads}')

        self.query = tf.keras.layers.Dense(units=config.hidden_size,
                                           kernel_initializer=create_initializer(config.initializer_range),
                                           name='query')
        self.key = tf.keras.layers.Dense(units=config.hidden_size,
                                         kernel_initializer=create_initializer(config.initializer_range), name='key')
        self.value = tf.keras.layers.Dense(units=config.hidden_size,
                                           kernel_initializer=create_initializer(config.initializer_range),
                                           name='value')
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.linear = tf.keras.layers.Dense(units=config.hidden_size,
                                            kernel_initializer=create_initializer(config.initializer_range))

    def call(self, inputs, attention_mask, batch_size, training=False):
        # B: Batch size, S: Sequence length, N: Num attention heads, H: Size per head
        query = self.query(inputs)  # (B, S_q, N * H)
        key = self.key(inputs)  # (B, S_k, N * H)
        value = self.value(inputs)  # Let S_k = S_v, (B, S_k, N * H)

        query = tf.reshape(query, [batch_size, -1, self.num_attention_heads, self.size_per_head])
        key = tf.reshape(key, [batch_size, -1, self.num_attention_heads, self.size_per_head])
        value = tf.reshape(value, [batch_size, -1, self.num_attention_heads, self.size_per_head])

        # (B, N, S_q, H) @ (B, N, S_k, H) -> (B, N, S_q, S_k)
        attention_scores = tf.einsum('bqnh,bknh->bnqk', query, key)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.size_per_head, dtype=attention_scores.dtype))

        if attention_mask is not None:
            attention_mask = tf.expand_dims(attention_mask, axis=[1])  # (B, 1, S_q, S_k)
            attention_scores += (1.0 - tf.cast(attention_mask, dtype=attention_scores.dtype)) * -10000.0

        attention_probs = tf.nn.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs, training=training)

        # (B, N, S_q, S_k) @ (B, S_k, N, H) -> (B, S_q, N, H)
        context = tf.einsum('bnqk,bknh->bqnh', attention_probs, value)
        context = tf.reshape(context, [batch_size, -1, self.num_attention_heads * self.size_per_head])

        output = self.linear(context)

        return output


class PositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig):
        super(PositionwiseFeedForward, self).__init__()
        self.intermediate = tf.keras.layers.Dense(units=config.intermediate_size,
                                                  kernel_initializer=create_initializer(config.initializer_range))
        self.gelu = tf.keras.layers.Activation(gelu)
        self.linear = tf.keras.layers.Dense(units=config.hidden_size,
                                            kernel_initializer=create_initializer(config.initializer_range))

    def call(self, inputs, training=False):
        return self.linear(self.gelu(self.intermediate(inputs)))
