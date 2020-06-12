import copy
import datetime
import os
import pickle
import random
import sys
import time
from typing import Dict

import torch
import torch.utils.data as data_utils
from IPython.core import ultratb
from torch.nn.utils.rnn import pad_sequence

import dataset
import train_lm
import train_mt
from albert_seq2seq import MassSeq2Seq
from lm import LM
from seq_gen import get_outputs_until_eos
from textprocessor import TextProcessor
from train_mt import MTTrainer

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


def mask_text(mask_prob, pads, texts, text_processor: TextProcessor):
    src_text = texts.clone()
    pad_indices = [int(pads.size(1)) - 1] * int(pads.size(0))
    pindices = torch.nonzero(~pads)
    for (r, c) in pindices:
        pad_indices[r] = min(pad_indices[r], int(c))
    pad_indices = torch.Tensor(pad_indices)

    mask_indices = [random.randint(1, int(x)) for x in pad_indices - (1 - mask_prob) * pad_indices]
    src_mask = torch.zeros(src_text.size(), dtype=torch.bool)
    to_recover = []
    to_recover_pos = []
    for i, mask_start in enumerate(mask_indices):
        src_mask[i, mask_start: mask_start + int(pad_indices[i] / 2)] = True
        to_recover.append(torch.cat([src_text[i, 0:1], src_text[i, mask_start: mask_start + int(pad_indices[i] / 2)]]))
        to_recover_pos.append(
            torch.cat([torch.arange(0, 1), torch.arange(mask_start, mask_start + int(pad_indices[i] / 2))]))
    to_recover = pad_sequence(to_recover, batch_first=True, padding_value=text_processor.pad_token_id())
    to_recover_pos = pad_sequence(to_recover_pos, batch_first=True, padding_value=int(src_text.size(-1)) - 1)

    assert 0 < mask_prob < 1
    tgt_mask = ~src_mask
    tgt_mask[~pads] = False  # We should not mask pads.
    tgt_mask[:, 0] = False  # Always unmask the first token (start symbol or language identifier).

    replacements = src_text[src_mask]
    for i in range(len(replacements)):
        r = random.random()
        if r < 0.8:
            replacements[i] = text_processor.mask_token_id()
        elif r < 0.9:
            # Replace with another random word.
            random_index = random.randint(len(text_processor.special_tokens), text_processor.vocab_size() - 1)
            replacements[i] = random_index
        else:
            # keep the word
            pass
    src_text[src_mask] = replacements
    masked_ids = texts[:, 1:][src_mask[:, 1:]]

    return src_mask, masked_ids, src_text, to_recover, to_recover_pos


class MassTrainer(MTTrainer):
    def train_epoch(self, data_iter: data_utils.DataLoader, dev_data_iter: data_utils.DataLoader, saving_path: str,
                    step: int, mt_dev_iter: data_utils.DataLoader = None, max_grad_norm: float = 1.0, **kwargs):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        sentences = 0
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        for i, batch in enumerate(data_iter):
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            src_inputs = batch["src_texts"].squeeze(0)
            src_pad_mask = src_inputs != model.text_processor.pad_token_id()

            src_mask, targets, src_text, to_recover, positions = mask_text(self.mask_prob, src_pad_mask, src_inputs,
                                                                           model.text_processor)

            if src_inputs.size(0) < self.num_gpu:
                continue

            try:
                predictions = self.model(device=self.device, src_inputs=src_text, tgt_inputs=to_recover,
                                         tgt_positions=positions, src_pads=src_pad_mask,
                                         pad_idx=model.text_processor.pad_token_id(),
                                         src_langs=batch["langs"].squeeze(0),
                                         log_softmax=True)
                ntokens = targets.size(0)

                if ntokens == 0:  # Nothing to predict!
                    continue

                loss = self.criterion(predictions, targets).mean() * ntokens
                loss.backward()

                loss = float(loss.data)
                total_loss += loss
                cur_loss += loss
                total_tokens += ntokens
                tokens += ntokens
                sentences += int(src_inputs.size(0))

                if self.optimizer is not None:
                    # We accumulate the gradients for both tasks!
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    step += 1

            except RuntimeError as err:
                torch.cuda.empty_cache()
                print("Error in processing", src_inputs.size(), src_inputs.size())

            if step % 50 == 0 and tokens > 0:
                elapsed = time.time() - start
                print(datetime.datetime.now(),
                      "Epoch Step: %d Loss: %f Tokens per Sec: %f Sentences per Sec: %f" % (
                          step, cur_loss / tokens, tokens / elapsed, sentences / elapsed))

                if step % 1000 == 0:
                    # Save every 1000 steps!
                    model.save(saving_path + ".latest")
                    with open(os.path.join(saving_path + ".latest", "optim"), "wb") as fp:
                        pickle.dump((self.optimizer, self.scheduler.last_epoch), fp)

                if step % 500 == 0:
                    self.validate(dev_data_iter)

                start, tokens, cur_loss, sentences = time.time(), 0, 0, 0

        print("Total loss in this epoch: %f" % (total_loss / total_tokens))
        model.save(saving_path + ".latest")
        with open(os.path.join(saving_path + ".latest", "optim"), "wb") as fp:
            pickle.dump((self.optimizer, self.scheduler.last_epoch), fp)

        self.validate(dev_data_iter)
        if mt_dev_iter is not None:
            bleu = self.eval_bleu(mt_dev_iter, saving_path)
            print("Pretraining BLEU:", bleu)
        return step

    def fine_tune(self, data_iter: data_utils.DataLoader, lang_directions: Dict[int, int], saving_path: str,
                  step: int, max_grad_norm: float = 1.0, dev_data_iter: data_utils.DataLoader = None):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        sentences = 0
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        for i, batch in enumerate(data_iter):
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            src_inputs = batch["src_texts"].squeeze(0)
            src_pad_mask = src_inputs != model.text_processor.pad_token_id()

            target_langs = torch.LongTensor([lang_directions[int(l)] for l in src_inputs[:, 0]])
            dst_langs = torch.LongTensor(
                [model.text_processor.languages[model.text_processor.id2token(lang_directions[int(l)])] for l in
                 src_inputs[:, 0]])
            if src_inputs.size(0) < self.num_gpu:
                continue

            try:
                model.eval()
                with torch.no_grad():
                    # We do not backpropagate the data generator following the MASS paper.
                    outputs = self.generator(device=self.device, src_inputs=src_inputs, first_tokens=target_langs,
                                             src_langs=batch["langs"].squeeze(0), tgt_langs=dst_langs,
                                             pad_idx=model.text_processor.pad_token_id(),
                                             src_mask=src_pad_mask, unpad_output=False)
                    if self.num_gpu > 1:
                        new_outputs = []
                        for output in outputs:
                            new_outputs += output
                        outputs = new_outputs

                    translations = pad_sequence(outputs, batch_first=True)
                    translation_pad_mask = (translations != model.text_processor.pad_token_id())
                model.train()

                # Now use it for back-translation loss.
                predictions = self.model(device=self.device, src_inputs=translations, tgt_inputs=src_inputs,
                                         src_pads=translation_pad_mask,
                                         pad_idx=model.text_processor.pad_token_id(),
                                         src_langs=dst_langs,
                                         tgt_langs=batch["langs"].squeeze(0),
                                         log_softmax=True)
                src_targets = src_inputs[:, 1:].contiguous().view(-1)
                src_mask_flat = src_pad_mask[:, 1:].contiguous().view(-1)
                targets = src_targets[src_mask_flat]
                ntokens = targets.size(0)

                if ntokens == 0:  # Nothing to predict!
                    continue

                bt_loss = self.criterion(predictions, targets).mean()
                bt_loss.backward()

                bt_loss = float(bt_loss.data) * ntokens
                total_loss += bt_loss
                cur_loss += bt_loss
                total_tokens += ntokens
                tokens += ntokens
                sentences += int(src_inputs.size(0))

                if self.optimizer is not None:
                    # We accumulate the gradients for both tasks!
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    step += 1

            except RuntimeError as err:
                torch.cuda.empty_cache()
                print("Error in processing", src_inputs.size(), src_inputs.size())

            if step % 50 == 0 and tokens > 0:
                elapsed = time.time() - start
                print(datetime.datetime.now(),
                      "Epoch Step: %d Loss: %f Tokens per Sec: %f Sentences per Sec: %f" % (
                          step, cur_loss / tokens, tokens / elapsed, sentences / elapsed))

                if step % 1000 == 0:
                    # Save every 1000 steps!
                    model.save(saving_path + ".beam.latest")
                    with open(os.path.join(saving_path + ".beam.latest", "optim"), "wb") as fp:
                        pickle.dump((self.optimizer, self.scheduler.last_epoch), fp)

                if step % 500 == 0 and dev_data_iter is not None:
                    bleu = self.eval_bleu(dev_data_iter, saving_path + ".beam")
                    print("BLEU:", bleu)

                start, tokens, cur_loss, sentences = time.time(), 0, 0, 0

        print("Total loss in this epoch: %f" % (total_loss / total_tokens))
        model.save(saving_path + ".beam.latest")
        with open(os.path.join(saving_path + ".latest", "optim"), "wb") as fp:
            pickle.dump((self.optimizer, self.scheduler.last_epoch), fp)

        if dev_data_iter is not None:
            bleu = self.eval_bleu(dev_data_iter, saving_path + ".beam")
            print("BLEU:", bleu)
        return step

    def validate(self, dev_data_iter):
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model.eval()
        with torch.no_grad():
            total_dev_loss, total_dev_tokens = 0, 0
            for batch in dev_data_iter:
                src_inputs = batch["src_texts"].squeeze(0)
                src_pad_mask = src_inputs != model.text_processor.pad_token_id()

                src_mask, targets, src_text, to_recover, positions = mask_text(self.mask_prob, src_pad_mask, src_inputs,
                                                                               model.text_processor)

                try:
                    predictions = self.model(device=self.device, src_inputs=src_text, tgt_inputs=to_recover,
                                             tgt_positions=positions, src_pads=src_pad_mask,
                                             pad_idx=model.text_processor.pad_token_id(),
                                             src_langs=batch["langs"].squeeze(0),
                                             log_softmax=True)
                    ntokens = targets.size(0)

                    if ntokens == 0:  # Nothing to predict!
                        continue

                    loss = self.criterion(predictions, targets).mean().data * ntokens
                    total_dev_loss += float(loss)
                    total_dev_tokens += ntokens
                except RuntimeError:
                    torch.cuda.empty_cache()
                    print("Error in processing", src_inputs.size(), src_inputs.size())

            dev_loss = total_dev_loss / total_dev_tokens
            print("Current dev loss", dev_loss)
            model.train()

    @staticmethod
    def train(options):
        if not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        if options.pretrained_path is not None:
            mt_model, lm = MassSeq2Seq.load(out_dir=options.pretrained_path, tok_dir=options.tokenizer_path,
                                            sep_decoder=options.sep_encoder)
            text_processor = mt_model.text_processor
        else:
            text_processor = TextProcessor(options.tokenizer_path)
            if options.lm_path is None:
                lm = LM(text_processor=text_processor, size=options.model_size)
            else:
                lm = LM.load(options.lm_path)

            decoder = copy.deepcopy(lm.encoder) if options.sep_encoder else lm.encoder
            mt_model = MassSeq2Seq(config=lm.config, encoder=lm.encoder, decoder=decoder, output_layer=lm.masked_lm,
                                   text_processor=lm.text_processor, checkpoint=options.checkpoint)
        MTTrainer.config_dropout(mt_model, options.dropout)

        pin_memory = torch.cuda.is_available()

        if options.continue_train:
            with open(os.path.join(options.pretrained_path, "optim"), "rb") as fp:
                optimizer, last_epoch = pickle.load(fp)
        else:
            optimizer, last_epoch = train_lm.LMTrainer.build_optimizer(mt_model, options.learning_rate,
                                                                       options.weight_decay), 0

        train_data, train_loader, dev_loader, finetune_loader, mt_dev_loader = None, None, None, None, None
        if options.step > 0 and last_epoch < options.step:
            train_data = dataset.MassDataset(batch_pickle_dir=options.train_path,
                                             max_batch_capacity=options.total_capacity, max_batch=options.batch,
                                             pad_idx=mt_model.text_processor.pad_token_id(),
                                             max_seq_len=options.max_seq_len, keep_examples=True)

            dev_data = dataset.MassDataset(batch_pickle_dir=options.dev_path,
                                           max_batch_capacity=options.total_capacity,
                                           max_batch=options.batch,
                                           pad_idx=mt_model.text_processor.pad_token_id(),
                                           max_seq_len=options.max_seq_len)
            train_loader = data_utils.DataLoader(train_data, batch_size=1, shuffle=True, pin_memory=pin_memory)
            dev_loader = data_utils.DataLoader(dev_data, batch_size=1, shuffle=False, pin_memory=pin_memory)

        lang_directions = {}
        if options.finetune_step > 0:
            finetune_data = dataset.MassDataset(batch_pickle_dir=options.train_path,
                                                max_batch_capacity=int(options.batch / (options.beam_width * 2)),
                                                max_batch=int(options.batch / (options.beam_width * 2)),
                                                pad_idx=mt_model.text_processor.pad_token_id(),
                                                max_seq_len=options.max_seq_len, keep_examples=False,
                                                example_list=None if train_data is None else train_data.examples_list)
            if train_data is not None:
                train_data.examples_list = []
            finetune_loader = data_utils.DataLoader(finetune_data, batch_size=1, shuffle=True, pin_memory=pin_memory)
            for lang1 in finetune_data.lang_ids:
                for lang2 in finetune_data.lang_ids:
                    if lang1 != lang2:
                        # Assuming that we only have two languages!
                        lang_directions[lang1] = lang2

        trainer = MassTrainer(model=mt_model, mask_prob=options.mask_prob, optimizer=optimizer, clip=options.clip,
                              warmup=options.warmup, step=options.step + options.finetune_step,
                              beam_width=options.beam_width, max_len_a=options.max_len_a, max_len_b=options.max_len_b,
                              len_penalty_ratio=options.len_penalty_ratio, last_epoch=last_epoch)

        mt_dev_loader = None
        if options.mt_dev_path is not None:
            mt_dev_data = dataset.MTDataset(batch_pickle_dir=options.mt_dev_path,
                                            max_batch_capacity=options.total_capacity,
                                            max_batch=int(options.batch / (options.beam_width * 2)),
                                            pad_idx=mt_model.text_processor.pad_token_id())
            mt_dev_loader = data_utils.DataLoader(mt_dev_data, batch_size=1, shuffle=False, pin_memory=pin_memory)

            print("creating reference")
            trainer.reference = []

            generator = (
                trainer.generator.module if hasattr(trainer.generator, "module") else trainer.generator
            )

            for batch in mt_dev_loader:
                tgt_inputs = batch["dst_texts"].squeeze()
                refs = get_outputs_until_eos(text_processor.sep_token_id(), tgt_inputs)
                ref = [generator.seq2seq_model.text_processor.tokenizer.decode(ref.numpy()) for ref in refs]
                trainer.reference += ref

        step, train_epoch = last_epoch, 1

        while options.step > 0 and step < options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(data_iter=train_loader, dev_data_iter=dev_loader,
                                       saving_path=options.model_path, mt_dev_iter=mt_dev_loader,
                                       step=step)
            train_epoch += 1

        finetune_epoch = 0
        mt_model.save(options.model_path + ".beam")
        while options.finetune_step > 0 and step <= options.finetune_step + options.step:
            print("finetune epoch", finetune_epoch)
            _ = trainer.fine_tune(data_iter=finetune_loader, lang_directions=lang_directions,
                                  saving_path=options.model_path, step=step, dev_data_iter=mt_dev_loader)
            finetune_epoch += 1


def get_option_parser():
    parser = train_mt.get_option_parser()
    parser.add_option("--dev_mt", dest="mt_dev_path",
                      help="Path to the MT dev data pickle files (SHOULD NOT BE USED IN UNSUPERVISED SETTING)",
                      metavar="FILE", default=None)
    parser.add_option("--fstep", dest="finetune_step", help="Number of finetuneing steps", type="int", default=125000)
    parser.set_default("mask_prob", 0.5)
    return parser


if __name__ == "__main__":
    parser = get_option_parser()
    (options, args) = parser.parse_args()
    print(options)
    MassTrainer.train(options=options)
    print("Finished Training!")
