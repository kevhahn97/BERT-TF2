import tensorflow as tf

from modeling.config import BertConfig
from modeling.utils import create_initializer, get_shape_list


class BertEmbedding(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig):
        super(BertEmbedding, self).__init__()
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.type_vocab_size = config.type_vocab_size

        self.word_embedding = self.add_weight(name='word_embedding',
                                              shape=[config.vocab_size, config.hidden_size],
                                              initializer=create_initializer(config.initializer_range))
        self.position_embedding = self.add_weight(name='position_embedding',
                                                  shape=[config.max_position_embeddings, config.hidden_size],
                                                  initializer=create_initializer(config.initializer_range))
        self.token_type_embedding = self.add_weight(name='token_type_embedding',
                                                    shape=[config.type_vocab_size, config.hidden_size],
                                                    initializer=create_initializer(config.initializer_range))

    def call(self, input_ids, token_type_ids, training=False):
        one_hot_input_ids = tf.one_hot(input_ids, depth=self.vocab_size)
        word_embedding = tf.matmul(one_hot_input_ids, self.word_embedding)

        batch_size, sequence_length, width = get_shape_list(word_embedding)

        one_hot_ids = tf.one_hot(token_type_ids, depth=self.type_vocab_size)
        token_type_embedding = tf.matmul(one_hot_ids, self.token_type_embedding)

        position_embedding = tf.slice(self.position_embedding, begin=[0, 0], size=[sequence_length, -1])
        position_embedding = tf.reshape(position_embedding, [1, sequence_length, width])

        embedding = word_embedding + token_type_embedding + position_embedding

        return embedding
