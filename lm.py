from transformers import AlbertModel, AlbertConfig

from textprocessor import TextProcessor


class LM:
    def __init__(self, text_processor: TextProcessor):
        self.text_processor: TextProcessor = text_processor
        self.lm: AlbertModel = AlbertModel(self._config(vocab_size=text_processor.tokenizer.get_vocab_size()))

    def _config(self, vocab_size: int) -> AlbertConfig:
        config = {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "embedding_size": 128,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 4,  # smaller than usual
            "num_hidden_layers": 4,  # smaller than usual
            "num_hidden_groups": 1,
            "net_structure_type": 0,
            "gap_size": 0,
            "num_memory_blocks": 0,
            "inner_group_num": 1,
            "down_scale_factor": 1,
            "type_vocab_size": 2,
            "vocab_size": vocab_size
        }

        return AlbertConfig(**config)