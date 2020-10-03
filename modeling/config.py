import copy
import json


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                 intermediate_size=3072, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512, type_vocab_size=16, initializer_range=0.02):
        """
        Constructs BertConfig.
        :param vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
        :param hidden_size: Size of the encoder layers and the pooler layer.
        :param num_hidden_layers: Number of hidden layers in the Transformer encoder.
        :param num_attention_heads: Number of attention heads for each attention layer in the Transformer encoder.
        :param intermediate_size: The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        :param hidden_dropout_prob: The dropout probability for all fully connected layers in the embeddings, encoder,
        :param attention_probs_dropout_prob: The dropout ratio for the attention probabilities.
        :param max_position_embeddings: The maximum sequence length that this model might ever be used with.
                Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        :param type_vocab_size: The vocabulary size of the `token_type_ids` passed into `BertModel`.
        :param initializer_range: The stdev of the truncated_normal_initializer for initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            return cls.from_dict(json.load(reader))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
