import copy
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig

from bert_seq2seq import BertEncoderModel, BertDecoderModel, BertOutputLayer, BertConfig
from textprocessor import TextProcessor


def decoder_config(vocab_size: int, pad_token_id: int, bos_token_id: int, eos_token_id: int, enc_layer: int = 6,
                   embed_dim: int = 768, intermediate_dim: int = 3072, num_lang: int = 1) -> PretrainedConfig:
    config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": embed_dim,
        "initializer_range": 0.02,
        "intermediate_size": intermediate_dim,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": enc_layer,
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "is_decoder": True,
        "type_vocab_size": num_lang,
    }
    return BertConfig(**config)


def future_mask(tgt_mask):
    attn_shape = (tgt_mask.size(0), tgt_mask.size(1), tgt_mask.size(1))
    future_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type_as(tgt_mask)
    return ~future_mask & tgt_mask.unsqueeze(-1)


class Seq2Seq(nn.Module):
    def __init__(self, text_processor: TextProcessor, lang_dec: bool = True,
                 enc_layer: int = 6, dec_layer: int = 3, embed_dim: int = 768, intermediate_dim: int = 3072):
        super(Seq2Seq, self).__init__()
        self.text_processor: TextProcessor = text_processor
        self.config = decoder_config(vocab_size=text_processor.tokenizer.get_vocab_size(),
                                     pad_token_id=text_processor.pad_token_id(),
                                     bos_token_id=text_processor.bos_token_id(),
                                     eos_token_id=text_processor.sep_token_id(),
                                     enc_layer=enc_layer, embed_dim=embed_dim, intermediate_dim=intermediate_dim,
                                     num_lang=len(text_processor.languages))

        self.enc_layer = enc_layer
        self.dec_layer = dec_layer
        self.embed_dim = embed_dim
        self.intermediate_dim = intermediate_dim

        dec_config = copy.deepcopy(self.config)
        dec_config.add_cross_attention = True
        dec_config.num_hidden_layers = self.dec_layer

        self.encoder = BertEncoderModel(self.config)
        self.encoder.init_weights()
        self.lang_dec = lang_dec
        self.decoder = BertDecoderModel(dec_config)
        if lang_dec:
            self.output_layer = nn.ModuleList([BertOutputLayer(dec_config) for _ in text_processor.languages])
        else:
            self.output_layer = BertOutputLayer(dec_config)

    def encode(self, src_inputs, src_mask, src_langs):
        device = self.encoder.embeddings.word_embeddings.weight.device
        if src_inputs.device != device:
            src_inputs = src_inputs.to(device)
            src_mask = src_mask.to(device)
            src_langs = src_langs.to(device)
        encoder_states = self.encoder(src_inputs, attention_mask=src_mask, token_type_ids=src_langs)
        return (encoder_states, None)

    def forward(self, src_inputs, tgt_inputs, src_mask, tgt_mask, src_langs, tgt_langs,
                log_softmax: bool = False):
        "Take in and process masked src and target sequences."
        device = self.encoder.embeddings.word_embeddings.weight.device
        batch_lang = int(tgt_langs[0])
        src_langs = src_langs.unsqueeze(-1).expand(-1, src_inputs.size(-1))
        tgt_langs = tgt_langs.unsqueeze(-1).expand(-1, tgt_inputs.size(-1)).to(device)
        src_inputs = src_inputs.to(device)
        src_langs = src_langs.to(device)
        if tgt_inputs.device != device:
            tgt_inputs = tgt_inputs.to(device)
            tgt_mask = tgt_mask.to(device)
        if src_mask.device != device:
            src_mask = src_mask.to(device)

        encoder_states = self.encode(src_inputs, src_mask, src_langs)[0]

        subseq_mask = future_mask(tgt_mask[:, :-1])
        if subseq_mask.device != tgt_inputs.device:
            subseq_mask = subseq_mask.to(device)

        output_layer = self.output_layer if not self.lang_dec else self.output_layer[batch_lang]

        decoder_output = self.decoder(encoder_states=encoder_states, input_ids=tgt_inputs[:, :-1],
                                 encoder_attention_mask=src_mask, tgt_attention_mask=subseq_mask,
                                 token_type_ids=tgt_langs[:, :-1])
        diag_outputs_flat = decoder_output.view(-1, decoder_output.size(-1))
        tgt_non_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
        non_padded_outputs = diag_outputs_flat[tgt_non_mask_flat]
        outputs = output_layer(non_padded_outputs)
        if log_softmax:
            outputs = F.log_softmax(outputs, dim=-1)
        return outputs

    def save(self, out_dir: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(os.path.join(out_dir, "mt_config"), "wb") as fp:
            pickle.dump(
                (self.lang_dec, self.enc_layer, self.dec_layer, self.embed_dim, self.intermediate_dim), fp)
        try:
            torch.save(self.state_dict(), os.path.join(out_dir, "mt_model.state_dict"))
        except:
            torch.cuda.empty_cache()
            torch.save(self.state_dict(), os.path.join(out_dir, "mt_model.state_dict"))
        finally:
            torch.cuda.empty_cache()

    @staticmethod
    def load(cls, out_dir: str, tok_dir: str):
        text_processor = TextProcessor(tok_model_path=tok_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(out_dir, "mt_config"), "rb") as fp:
            lang_dec, enc_layer, dec_layer, embed_dim, intermediate_dim = pickle.load(fp)

            mt_model = cls(text_processor=text_processor, lang_dec=lang_dec, enc_layer=enc_layer,
                           dec_layer=dec_layer, embed_dim=embed_dim, intermediate_dim=intermediate_dim)

            mt_model.load_state_dict(torch.load(os.path.join(out_dir, "mt_model.state_dict"), map_location=device),
                                     strict=False)
            return mt_model
