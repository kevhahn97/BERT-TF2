import tensorflow as tf

from modeling.config import BertConfig
from modeling.layers.bert import Bert
from modeling.layers.pretrain import MaskedLanguageModel, NextSentencePrediction


class BertPretrain(tf.keras.Model):
    def __init__(self, config):
        super(BertPretrain, self).__init__()
        self.bert = Bert(config=config)
        self.masked_language_model = MaskedLanguageModel(config=config,
                                                         word_embedding=self.bert.bert_embedding.word_embedding)
        self.next_sentence_prediction = NextSentencePrediction(config=config)

    def call(self, inputs, training=False):
        # parse inputs
        if isinstance(inputs, dict):
            input_ids = inputs.get('input_ids')
            input_mask = inputs.get('input_mask')
            token_type_ids = inputs.get('token_type_ids')
            mask_lm_positions = inputs.get('mask_lm_positions')
            assert input_ids is not None
            assert mask_lm_positions is not None
        else:
            raise NotImplementedError

        context = self.bert(input_ids, input_mask, token_type_ids, training=training)
        pretrain_logits = self.masked_language_model(context, mask_lm_positions), self.next_sentence_prediction(context)
        return pretrain_logits

    def run_dummy_inputs(self, batch_size=4, max_seq_len=512):
        inputs = dict(input_ids=tf.zeros([batch_size, max_seq_len], dtype=tf.int32),
                      mask_lm_positions=tf.random.uniform(shape=[batch_size, 32], minval=0, maxval=max_seq_len,
                                                          dtype=tf.int32))
        return self(inputs)


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    vocab_size = 3000

    m1 = BertPretrain(BertConfig(vocab_size=vocab_size))
    m1.run_dummy_inputs()
    m1.save_weights('t.h5')

    m2 = BertPretrain(BertConfig(vocab_size=vocab_size))
    m2.run_dummy_inputs()
    m2.load_weights('t.h5')

    for m1v, m2v in zip(m1.trainable_variables, m2.trainable_variables):
        assert tf.equal(m1v, m2v).numpy().all()

    exit(0)
