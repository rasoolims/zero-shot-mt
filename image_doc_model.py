import copy
import pickle

import torch.nn.functional as F
from torchvision import models
from transformers.modeling_albert import *

from albert_seq2seq import MassSeq2Seq, future_mask, AlbertDecoderTransformer
from lm import LM
from textprocessor import TextProcessor


class ModifiedResnet(models.ResNet):
    def _forward_impl(self, x):
        input = x
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        grid_hidden = self.layer4(x)
        grid_hidden = grid_hidden.view(grid_hidden.size(0), grid_hidden.size(1), -1)
        grid_hidden = grid_hidden.permute((0, 2, 1))
        if self.dropout > 0:
            grid_hidden = F.dropout(grid_hidden, p=self.dropout)
        grid_outputs = F.relu(self.fc(grid_hidden))
        location_embedding = self.location_embedding.weight.unsqueeze(0)
        out = grid_outputs + location_embedding
        out_norm = self.layer_norm(out)
        if self.dropout > 0:
            out_norm = F.dropout(out_norm, p=self.dropout)
        return out_norm


def init_net(embed_dim: int, dropout: float = 0.1, freeze: bool = False, depth: int = 1):
    if depth == 1:
        model = models.resnet18(pretrained=True)
    elif depth == 2:
        model = models.resnet34(pretrained=True)
    elif depth == 3:
        model = models.resnet50(pretrained=True)
    elif depth == 4:
        model = models.resnet101(pretrained=True)
    elif depth == 5:
        model = models.resnet152(pretrained=True)

    model.__class__ = ModifiedResnet
    model.dropout = dropout
    model.layer_norm = torch.nn.LayerNorm(embed_dim, eps=1e-12)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    current_weight = model.state_dict()["fc.weight"]
    model.fc = torch.nn.Linear(in_features=current_weight.size()[1], out_features=embed_dim, bias=False)
    model.fc.train()

    # Learning embedding of each CNN region.
    model.location_embedding = nn.Embedding(49, embed_dim)
    model.location_embedding.train(True)

    return model


class ImageMassSeq2Seq(MassSeq2Seq):
    def __init__(self, config: AlbertConfig, encoder: AlbertModel, decoder, output_layer: AlbertMLMHead,
                 text_processor: TextProcessor, checkpoint: int = 5, freeze_image: bool = False,
                 resnet_depth: int = 1, lang_dec: bool = False, num_cross_layers: int = None):
        super(ImageMassSeq2Seq, self).__init__(config, encoder, decoder, output_layer, text_processor, lang_dec,
                                               checkpoint)
        self.image_model: ModifiedResnet = init_net(embed_dim=config.embedding_size, dropout=config.hidden_dropout_prob,
                                                    freeze=freeze_image, depth=resnet_depth)
        if num_cross_layers is None:
            self.image_self_attention = AlbertTransformer(config)
            self.cross_decoder = AlbertDecoderTransformer(AlbertTransformer(config))
        else:
            cross_config = copy.deepcopy(config)
            cross_config.num_hidden_layers = num_cross_layers
            self.cross_decoder = AlbertDecoderTransformer(AlbertTransformer(cross_config))
            self.image_self_attention = AlbertTransformer(cross_config)
        self.multimodal_attention_gate = nn.Parameter(torch.zeros(1, config.hidden_size).fill_(0.1), requires_grad=True)
        self.back_mapper = nn.Linear(config.hidden_size, config.embedding_size)

    def encode(self, src_inputs, src_mask, src_langs, images=None):
        encoder_states = super().encode(src_inputs, src_mask, src_langs)
        if images is not None:
            device = self.encoder.embeddings.word_embeddings.weight.device
            if isinstance(images, list):
                images = images[0]
            if images.device != device:
                images = images.to(device)
            image_embeddings = self.image_model(images)
            head_mask = [None] * self.image_self_attention.config.num_hidden_layers
            image_attended = self.image_self_attention(hidden_states=image_embeddings, head_mask=head_mask)[0]
            return encoder_states[0], image_attended
        return encoder_states

    def forward(self, src_inputs=None, src_pads=None, tgt_inputs=None, src_langs=None, tgt_langs=None, pad_idx: int = 1,
                tgt_positions=None, batch=None, log_softmax: bool = False, **kwargs):
        if isinstance(batch, list):
            assert len(batch) == 1
            batch = batch[0]
            tgt_inputs = tgt_inputs[0]
            src_langs = src_langs[0]
        if isinstance(src_pads, list):
            src_pads = src_pads[0]
        if isinstance(src_inputs, list):
            src_inputs = src_inputs[0]
        if isinstance(tgt_positions, list):
            tgt_positions = tgt_positions[0]

        if batch is None:
            return super().forward(src_inputs=src_inputs, src_pads=src_pads, tgt_inputs=tgt_inputs, src_langs=src_langs,
                                   tgt_langs=tgt_langs, pad_idx=pad_idx, tgt_positions=tgt_positions,
                                   log_softmax=log_softmax)

        assert src_inputs is not None
        assert tgt_inputs is not None

        device = self.encoder.embeddings.word_embeddings.weight.device
        images = batch["images"].to(device)

        tgt_inputs = tgt_inputs.to(device)
        tgt_mask = tgt_inputs != pad_idx
        src_pads = src_pads.to(device)
        src_inputs = src_inputs.to(device)
        src_langs_t = src_langs.unsqueeze(-1).expand(-1, src_inputs.size(-1)).to(device)
        batch_lang = int(src_langs[0])

        decoder = self.decoder if not self.lang_dec else self.decoder[batch_lang]
        output_layer = self.output_layer if not self.lang_dec else self.output_layer[batch_lang]
        tgt_langs = src_langs.unsqueeze(-1).expand(-1, tgt_inputs.size(-1)).to(device)
        if tgt_positions is not None:
            tgt_positions = tgt_positions[:, :-1].to(device)

        subseq_mask = future_mask(tgt_mask[:, :-1])

        encoder_states, image_attended = self.encode(src_inputs, src_pads, src_langs_t, images)

        text_decoder_output = decoder(encoder_states=encoder_states, input_ids=tgt_inputs[:, :-1],
                                 input_ids_mask=tgt_mask[:, :-1], src_attn_mask=src_pads,
                                 tgt_attn_mask=subseq_mask,
                                 position_ids=tgt_positions,
                                 token_type_ids=tgt_langs[:, :-1])
        image_decoder_output = decoder(encoder_states=image_attended, input_ids=tgt_inputs[:, :-1],
                                      input_ids_mask=tgt_mask[:, :-1],
                                      tgt_attn_mask=subseq_mask,
                                      position_ids=tgt_positions,
                                      token_type_ids=tgt_langs[:, :-1])
        eps = 1e-7
        sig_gate = torch.sigmoid(self.multimodal_attention_gate + eps)
        decoder_output = sig_gate * text_decoder_output + (1 - sig_gate) * image_decoder_output


        diag_outputs_flat = decoder_output.view(-1, decoder_output.size(-1))
        tgt_non_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
        non_padded_outputs = diag_outputs_flat[tgt_non_mask_flat]
        outputs = output_layer(non_padded_outputs)
        if log_softmax:
            outputs = F.log_softmax(outputs, dim=-1)
        return outputs

    @staticmethod
    def load(out_dir: str, tok_dir: str, sep_decoder: bool, resnet_depth: int = 1, lang_dec: bool = False):
        text_processor = TextProcessor(tok_model_path=tok_dir)
        with open(os.path.join(out_dir, "mt_config"), "rb") as fp:
            config, checkpoint = pickle.load(fp)
            lm = LM(text_processor=text_processor, config=config)
            decoder = copy.deepcopy(lm.encoder) if sep_decoder else lm.encoder
            mt_model = ImageMassSeq2Seq(config=config, encoder=lm.encoder, decoder=decoder,
                                        output_layer=lm.masked_lm, resnet_depth=resnet_depth,
                                        text_processor=lm.text_processor, checkpoint=checkpoint, lang_dec=lang_dec)
            mt_model.load_state_dict(torch.load(os.path.join(out_dir, "mt_model.state_dict")))
            return mt_model, lm
