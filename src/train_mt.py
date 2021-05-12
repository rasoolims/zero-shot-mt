import datetime
import pickle
import sys
import time
from itertools import chain
from typing import List

import sacrebleu
import torch.nn as nn
import torch.utils.data as data_utils
from IPython.core import ultratb
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import dataset
from loss import SmoothedNLLLoss
from option_parser import get_mt_options_parser
from seq2seq import Seq2Seq
from seq_gen import BeamDecoder, get_outputs_until_eos
from utils import *

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class Trainer:
    def __init__(self, model, mask_prob: float = 0.3, clip: int = 1, optimizer=None,
                 beam_width: int = 5, max_len_a: float = 1.1, max_len_b: int = 5, len_penalty_ratio: float = 0.8,
                 nll_loss: bool = False, fp16: bool = False, rank: int = -1):
        self.model = model

        self.clip = clip
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpu = torch.cuda.device_count()

        self.mask_prob = mask_prob
        if nll_loss:
            self.criterion = nn.NLLLoss(ignore_index=model.text_processor.pad_token_id())
        else:
            self.criterion = SmoothedNLLLoss(ignore_index=model.text_processor.pad_token_id())

        self.num_gpu = torch.cuda.device_count()
        self.fp16 = False
        self.rank = rank
        if rank >= 0:
            self.device = torch.device('cuda', rank)
            torch.cuda.set_device(self.device)

        self.model = self.model.to(self.device)

        if fp16:
            # todo
            self.fp16 = True

        self.generator = BeamDecoder(self.model, beam_width=beam_width, max_len_a=max_len_a, max_len_b=max_len_b,
                                     len_penalty_ratio=len_penalty_ratio)
        if rank >= 0:
            self.model = DistributedDataParallel(self.model, device_ids=[self.rank], output_device=self.rank,
                                                 find_unused_parameters=True)
            self.generator = DistributedDataParallel(self.generator, device_ids=[self.rank], output_device=self.rank,
                                                     find_unused_parameters=True)

        self.reference = None
        self.best_bleu = -1.0

    def train_epoch(self, step: int, saving_path: str = None,
                    mass_data_iter: List[data_utils.DataLoader] = None, mt_dev_iter: List[data_utils.DataLoader] = None,
                    mt_train_iter: List[data_utils.DataLoader] = None, max_step: int = 300000, accum=1,
                    save_opt: bool = False,
                    **kwargs):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        batch_zip, shortest = self.get_batch_zip(mass_data_iter, mt_train_iter)

        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        self.optimizer.zero_grad()
        for i, batches in enumerate(batch_zip):
            for batch in batches:
                is_mass_batch = "dst_texts" not in batch
                try:
                    if not is_mass_batch:  # MT data
                        src_inputs = batch["src_texts"].squeeze(0)
                        src_mask = batch["src_pad_mask"].squeeze(0)
                        tgt_inputs = batch["dst_texts"].squeeze(0)
                        tgt_mask = batch["dst_pad_mask"].squeeze(0)
                        dst_langs = batch["dst_langs"].squeeze(0)
                        if src_inputs.size(0) < self.num_gpu:
                            continue
                        predictions = self.model(src_inputs=src_inputs, tgt_inputs=tgt_inputs, src_mask=src_mask,
                                                 tgt_mask=tgt_mask, tgt_langs=dst_langs, log_softmax=True)
                        targets = tgt_inputs[:, 1:].contiguous().view(-1)
                        tgt_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
                        targets = targets[tgt_mask_flat]
                        ntokens = targets.size(0)
                    else:  # MASS data
                        src_inputs = batch["src_texts"].squeeze(0)
                        pad_indices = batch["pad_idx"].squeeze(0)
                        if src_inputs.size(0) < self.num_gpu:
                            continue

                        masked_info = mass_mask(self.mask_prob, pad_indices, src_inputs, model.text_processor)
                        predictions = self.model(src_inputs=masked_info["src_text"],
                                                 tgt_inputs=masked_info["to_recover"],
                                                 tgt_positions=masked_info["positions"],
                                                 src_langs=batch["langs"].squeeze(0),
                                                 log_softmax=True)
                        targets = masked_info["targets"]
                        ntokens = targets.size(0)

                    if self.num_gpu == 1:
                        targets = targets.to(predictions.device)
                    if self.rank >= 0: targets = targets.to(self.device)

                    loss = self.criterion(predictions, targets).mean()
                    backward(loss, self.optimizer, self.fp16)

                    loss = float(loss.data) * ntokens
                    tokens += ntokens
                    total_tokens += ntokens
                    total_loss += loss
                    cur_loss += loss

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    step += 1
                    if step % accum == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if is_mass_batch:
                        mass_unmask(masked_info["src_text"], masked_info["src_mask"], masked_info["mask_idx"])

                    if step % 50 == 0 and tokens > 0:
                        elapsed = time.time() - start
                        print(self.rank, "->", datetime.datetime.now(),
                              "Epoch Step: %d Loss: %f Tokens per Sec: %f " % (
                                  step, cur_loss / tokens, tokens / elapsed))

                        if mt_dev_iter is not None and step % 5000 == 0 and self.rank <= 0:
                            bleu = self.eval_bleu(mt_dev_iter, saving_path)
                            print("BLEU:", bleu)

                        if step % 10000 == 0:
                            if self.rank <= 0:
                                if self.rank < 0:
                                    model.cpu().save(saving_path + ".latest")
                                elif self.rank == 0:
                                    model.save(saving_path + ".latest")

                                if save_opt:
                                    with open(os.path.join(saving_path + ".latest", "optim"), "wb") as fp:
                                        pickle.dump(self.optimizer, fp)
                                if self.rank < 0:
                                    model = model.to(self.device)

                        start, tokens, cur_loss = time.time(), 0, 0

                except RuntimeError as err:
                    print(repr(err))
                    torch.cuda.empty_cache()

            if i == shortest - 1:
                break
            if step >= max_step:
                break

        try:
            if self.rank <= 0:
                print("Total loss in this epoch: %f" % (total_loss / total_tokens))
                if self.rank < 0:
                    model.cpu().save(saving_path + ".latest")
                    model = model.to(self.device)
                elif self.rank == 0:
                    model.save(saving_path + ".latest")

                if mt_dev_iter is not None:
                    bleu = self.eval_bleu(mt_dev_iter, saving_path)
                    print("BLEU:", bleu)
        except RuntimeError as err:
            print(repr(err))

        return step

    def get_batch_zip(self, mass_data_iter, mt_train_iter):
        iters = list(chain(*filter(lambda x: x != None, [mass_data_iter, mt_train_iter])))
        shortest = min(len(l) for l in iters)
        return zip(*iters), shortest

    def eval_bleu(self, dev_data_iter, saving_path, save_opt: bool = False):
        mt_output = []
        src_text = []
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model.eval()

        with torch.no_grad():
            for iter in dev_data_iter:
                for batch in iter:
                    src_inputs = batch["src_texts"].squeeze(0)
                    src_mask = batch["src_pad_mask"].squeeze(0)
                    tgt_inputs = batch["dst_texts"].squeeze(0)
                    dst_langs = batch["dst_langs"].squeeze(0)
                    src_pad_idx = batch["src_pad_idx"].squeeze(0)

                    src_ids = get_outputs_until_eos(model.src_eos_id(), src_inputs, remove_first_token=True)
                    src_text += list(map(lambda src: model.decode_src(src), src_ids))

                    outputs = self.generator(src_inputs=src_inputs, src_sizes=src_pad_idx,
                                             first_tokens=tgt_inputs[:, 0],
                                             src_mask=src_mask, tgt_langs=dst_langs,
                                             pad_idx=model.text_processor.pad_token_id())
                    if self.num_gpu > 1 and self.rank < 0:
                        new_outputs = []
                        for output in outputs:
                            new_outputs += output
                        outputs = new_outputs

                    mt_output += list(map(lambda x: model.text_processor.tokenizer.decode(x[1:].numpy()), outputs))

            model.train()
        bleu = sacrebleu.corpus_bleu(mt_output, [self.reference[:len(mt_output)]], lowercase=True, tokenize="intl")

        with open(os.path.join(saving_path, "bleu.output"), "w") as writer:
            writer.write("\n".join(
                [src + "\n" + ref + "\n" + o + "\n\n***************\n" for src, ref, o in
                 zip(src_text, mt_output, self.reference[:len(mt_output)])]))

        if bleu.score > self.best_bleu:
            self.best_bleu = bleu.score
            print("Saving best BLEU", self.best_bleu)
            with open(os.path.join(saving_path, "bleu.best.output"), "w") as writer:
                writer.write("\n".join(
                    [src + "\n" + ref + "\n" + o + "\n\n***************\n" for src, ref, o in
                     zip(src_text, mt_output, self.reference[:len(mt_output)])]))
            if self.rank < 0:
                model.cpu().save(saving_path)
                model = model.to(self.device)
            elif self.rank == 0:
                model.save(saving_path)

            if save_opt:
                with open(os.path.join(saving_path, "optim"), "wb") as fp:
                    pickle.dump(self.optimizer, fp)

        return bleu.score

    @staticmethod
    def train(options):
        if options.local_rank <= 0 and not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        text_processor = TextProcessor(options.tokenizer_path)

        assert text_processor.pad_token_id() == 0
        num_processors = max(torch.cuda.device_count(), 1) if options.local_rank < 0 else 1

        if options.pretrained_path is not None:
            mt_model = Seq2Seq.load(Seq2Seq, options.pretrained_path, tok_dir=options.tokenizer_path)
        else:
            mt_model = Seq2Seq(text_processor=text_processor, lang_dec=options.lang_decoder,
                               dec_layer=options.decoder_layer, embed_dim=options.embed_dim,
                               intermediate_dim=options.intermediate_layer_dim, freeze_encoder=options.freeze_encoder,
                               shallow_encoder=options.shallow_encoder)

        print("Model initialization done!")

        if options.continue_train:
            with open(os.path.join(options.pretrained_path, "optim"), "rb") as fp:
                optimizer = pickle.load(fp)
        else:
            optimizer = build_optimizer(mt_model, options.learning_rate, warump_steps=options.warmup)
        trainer = Trainer(model=mt_model, mask_prob=options.mask_prob, optimizer=optimizer, clip=options.clip,
                          beam_width=options.beam_width, max_len_a=options.max_len_a,
                          max_len_b=options.max_len_b, len_penalty_ratio=options.len_penalty_ratio,
                          fp16=options.fp16, rank=options.local_rank)

        pin_memory = torch.cuda.is_available()

        mass_train_data, mass_train_loader, mt_dev_loader = None, None, None
        if options.mass_train_path is not None:
            mass_train_paths = options.mass_train_path.strip().split(",")
            if options.step > 0:
                mass_train_data, mass_train_loader = Trainer.get_mass_loader(mass_train_paths, mt_model,
                                                                             num_processors, options,
                                                                             pin_memory,
                                                                             keep_examples=False)

        mt_train_loader = None
        if options.mt_train_path is not None:
            mt_train_loader = Trainer.get_mt_train_data(mt_model, num_processors, options, pin_memory)

        mt_dev_loader = None
        if options.mt_dev_path is not None:
            mt_dev_loader = Trainer.get_mt_dev_data(mt_model, options, pin_memory, text_processor, trainer)

        step, train_epoch = 0, 1
        while options.step > 0 and step < options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(mass_data_iter=mass_train_loader,
                                       mt_train_iter=mt_train_loader, max_step=options.step,
                                       mt_dev_iter=mt_dev_loader, saving_path=options.model_path, step=step,
                                       save_opt=options.save_opt, accum=options.accum)
            train_epoch += 1

    @staticmethod
    def get_mt_dev_data(mt_model, options, pin_memory, text_processor, trainer):
        mt_dev_loader = []
        dev_paths = options.mt_dev_path.split(",")
        trainer.reference = []
        for dev_path in dev_paths:
            mt_dev_data = dataset.MTDataset(batch_pickle_dir=dev_path,
                                            max_batch_capacity=options.total_capacity, keep_src_pad_idx=True,
                                            max_batch=int(options.batch / (options.beam_width * 2)),
                                            src_pad_idx=mt_model.src_pad_id(),
                                            dst_pad_idx=mt_model.text_processor.pad_token_id())
            dl = data_utils.DataLoader(mt_dev_data, batch_size=1, shuffle=False, pin_memory=pin_memory)
            mt_dev_loader.append(dl)

            print(options.local_rank, "creating reference")

            generator = (
                trainer.generator.module if hasattr(trainer.generator, "module") else trainer.generator
            )

            for batch in dl:
                tgt_inputs = batch["dst_texts"].squeeze()
                refs = get_outputs_until_eos(text_processor.sep_token_id(), tgt_inputs, remove_first_token=True)
                ref = [generator.seq2seq_model.text_processor.tokenizer.decode(ref.numpy()) for ref in refs]
                trainer.reference += ref
        return mt_dev_loader

    @staticmethod
    def get_mt_train_data(mt_model, num_processors, options, pin_memory: bool):
        mt_train_loader = []
        train_paths = options.mt_train_path.split(",")
        for train_path in train_paths:
            mt_train_data = dataset.MTDataset(batch_pickle_dir=train_path,
                                              max_batch_capacity=int(num_processors * options.total_capacity / 2),
                                              max_batch=int(num_processors * options.batch / 2),
                                              src_pad_idx=mt_model.src_pad_id(),
                                              dst_pad_idx=mt_model.text_processor.pad_token_id(),
                                              keep_src_pad_idx=False)
            mtl = data_utils.DataLoader(mt_train_data,
                                        sampler=None if options.local_rank < 0 else DistributedSampler(mt_train_data,
                                                                                                       rank=options.local_rank),
                                        batch_size=1, shuffle=(options.local_rank < 0), pin_memory=pin_memory)
            mt_train_loader.append(mtl)
        return mt_train_loader

    @staticmethod
    def get_mass_loader(mass_train_paths, mt_model, num_processors, options, pin_memory, keep_examples):
        mass_train_data, mass_train_loader = [], []
        for i, mass_train_path in enumerate(mass_train_paths):
            td = dataset.MassDataset(batch_pickle_dir=mass_train_path,
                                     max_batch_capacity=num_processors * options.total_capacity,
                                     max_batch=num_processors * options.batch,
                                     src_pad_idx=mt_model.src_pad_id(),
                                     max_seq_len=options.max_seq_len, keep_examples=keep_examples, )
            mass_train_data.append(td)

            dl = data_utils.DataLoader(td, sampler=None if options.local_rank < 0 else DistributedSampler(td,
                                                                                                          rank=options.local_rank),
                                       batch_size=1,
                                       shuffle=(options.local_rank < 0), pin_memory=pin_memory)
            mass_train_loader.append(dl)
        return mass_train_data, mass_train_loader


if __name__ == "__main__":
    parser = get_mt_options_parser()
    (options, args) = parser.parse_args()
    if options.local_rank <= 0:
        print(options)
    init_distributed(options)
    Trainer.train(options=options)
    if options.local_rank >= 0: cleanup_distributed(options)
    print("Finished Training!")
