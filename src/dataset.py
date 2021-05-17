import datetime
import glob
import logging
import marshal
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MTDataset(Dataset):
    def __init__(self, max_batch_capacity: int, max_batch: int,
                 src_pad_idx: int, dst_pad_idx: int, max_seq_len: int = 175, batch_pickle_dir: str = None,
                 examples: List[Tuple[torch.tensor, torch.tensor, int, int]] = None, keep_src_pad_idx=True,
                 ngpu=1):
        self.keep_src_pad_idx = keep_src_pad_idx
        self.ngpu = ngpu

        if examples is None:
            self.build_batches(batch_pickle_dir, max_batch_capacity, max_batch, src_pad_idx, dst_pad_idx, max_seq_len)
        else:
            self.batch_examples(examples, max_batch, max_batch_capacity, max_seq_len, ngpu, src_pad_idx, dst_pad_idx)

    def build_batches(self, batch_pickle_dir: str, max_batch_capacity: int, max_batch: int,
                      src_pad_idx: int, dst_pad_idx: int, max_seq_len: int = 175):
        """
        Since training is fully-batched and has memory/computational need for cubic power of target length, and quadratic
        power of source length, we need to make sure that each batch has similar length and it does not go over
        max_batch_capacity. We also need to make sure not to include those batches that has less than num_gpu
        sentence pairs (it will crash in multi-gpu).
        """
        with open(batch_pickle_dir, "rb") as fr:
            print("LOADING MT BATCHES")
            examples: List[Tuple[torch.tensor, torch.tensor, int, int]] = marshal.load(fr)
            self.batch_examples(examples, max_batch, max_batch_capacity, max_seq_len, self.ngpu, src_pad_idx,
                                dst_pad_idx)

    def batch_examples(self, examples, max_batch, max_batch_capacity, max_seq_len, num_gpu, src_pad_idx, dst_pad_idx):
        print("BUILDING MT BATCHES")
        self.batches = []
        cur_src_batch, cur_srct_batch, cur_dst_batch = [], [], []
        cur_max_src_len, cur_max_srct_len, cur_max_dst_len = 0, 0, 0
        cur_dst_langs, cur_lex_cand_batch = [], []
        for ei, example in enumerate(examples):
            src = torch.LongTensor(example[0][:max_seq_len])  # trim if longer than expected!
            dst = torch.LongTensor(example[1][:max_seq_len])  # trim if longer than expected!
            srct = torch.LongTensor(example[3][:max_seq_len])  # trim if longer than expected!
            cur_dst_langs.append(example[2])
            cur_max_src_len = max(cur_max_src_len, int(src.size(0)))
            cur_max_srct_len = max(cur_max_srct_len, int(srct.size(0)))
            cur_max_dst_len = max(cur_max_dst_len, int(dst.size(0)))

            cur_src_batch.append(src)
            cur_srct_batch.append(srct)
            cur_dst_batch.append(dst)

            batch_capacity_size = (cur_max_src_len ** 2 + cur_max_dst_len ** 2) * len(
                cur_src_batch) * cur_max_dst_len
            batch_size = (cur_max_src_len + cur_max_dst_len) * len(cur_src_batch)

            if (batch_size > max_batch or batch_capacity_size > max_batch_capacity * 1000000) and \
                    len(cur_src_batch[:-1]) >= num_gpu and len(cur_src_batch) > 1:
                src_batch = pad_sequence(cur_src_batch[:-1], batch_first=True, padding_value=src_pad_idx)
                # Bellow should be dst_pad_idx since srct has the same tokenizer as the target output.
                srct_batch = pad_sequence(cur_srct_batch[:-1], batch_first=True, padding_value=dst_pad_idx)
                dst_batch = pad_sequence(cur_dst_batch[:-1], batch_first=True, padding_value=dst_pad_idx)
                src_pad_mask = (src_batch != src_pad_idx)
                # Bellow should be dst_pad_idx since srct has the same tokenizer as the target output.
                srct_pad_mask = (src_batch != dst_pad_idx)
                dst_pad_mask = (dst_batch != dst_pad_idx)

                entry = {"src_texts": src_batch, "srct_texts": srct_batch, "src_pad_mask": src_pad_mask,
                         "dst_texts": dst_batch,
                         "srct_pad_mask": srct_pad_mask, "dst_pad_mask": dst_pad_mask,
                         "dst_langs": torch.LongTensor(cur_dst_langs[:-1])}
                self.batches.append(entry)
                cur_src_batch, cur_src_batch, cur_dst_batch = [cur_src_batch[-1]], [cur_srct_batch[-1]], [
                    cur_dst_batch[-1]]
                cur_dst_langs = [cur_dst_langs[-1]]
                cur_max_src_len, cur_max_srct_len, cur_max_dst_len = int(cur_src_batch[0].size(0)), int(
                    cur_srct_batch[0].size(0)), int(cur_dst_batch[0].size(0))

            if ei % 1000 == 0:
                print(ei, "/", len(examples), end="\r")

        if len(cur_src_batch) > 0 and len(cur_src_batch) >= num_gpu:
            src_batch = pad_sequence(cur_src_batch, batch_first=True, padding_value=src_pad_idx)
            # Bellow should be dst_pad_idx since srct has the same tokenizer as the target output.
            srct_batch = pad_sequence(cur_srct_batch, batch_first=True, padding_value=dst_pad_idx)
            dst_batch = pad_sequence(cur_dst_batch, batch_first=True, padding_value=dst_pad_idx)
            src_pad_mask = (src_batch != src_pad_idx)
            # Bellow should be dst_pad_idx since srct has the same tokenizer as the target output.
            srct_pad_mask = (srct_batch != dst_pad_idx)
            dst_pad_mask = (dst_batch != dst_pad_idx)
            entry = {"src_texts": src_batch, "src_pad_mask": src_pad_mask, "dst_texts": dst_batch,
                     "srct_texts": srct_batch, "srct_pad_mask": srct_pad_mask,
                     "dst_pad_mask": dst_pad_mask, "dst_langs": torch.LongTensor(cur_dst_langs)}
            self.batches.append(entry)

        if self.keep_src_pad_idx:
            src_pad_idx_find = lambda non_zeros, sz: (sz - 1) if int(non_zeros.size(0)) == 0 else non_zeros[0]
            for b in self.batches:
                pads = b["src_texts"] == src_pad_idx
                sz = int(pads.size(1))
                pad_indices = torch.LongTensor(list(map(lambda p: src_pad_idx_find(torch.nonzero(p), sz), pads)))
                b["src_pad_idx"] = pad_indices

        print("\nLoaded %d bitext sentences to %d batches!" % (len(examples), len(self.batches)))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        return self.batches[item]


class MassDataset(Dataset):
    def __init__(self, batch_pickle_dir: str, max_batch_capacity: int, max_batch: int,
                 src_pad_idx: int, max_seq_len: int = 512, keep_examples: bool = False, example_list: List = None,
                 keep_src_pad_idx=True, ngpu=1):
        self.keep_src_pad_idx = keep_src_pad_idx
        self.ngpu = ngpu
        if example_list is None:
            self.build_batches(batch_pickle_dir, max_batch_capacity, max_batch, src_pad_idx, max_seq_len, keep_examples)
        else:
            self.examples_list = example_list
            self.batch_items(max_batch, max_batch_capacity, max_seq_len, src_pad_idx)

    @staticmethod
    def read_example_file(path):
        print(datetime.datetime.now(), "Loading", path)
        with open(path, "rb") as fr:
            examples: List[Tuple[torch.tensor, torch.tensor]] = marshal.load(fr)
        return examples

    def build_batches(self, batch_pickle_dir: str, max_batch_capacity: int, max_batch: int,
                      src_pad_idx: int, max_seq_len: int = 175, keep_examples: bool = False):
        """
        Since training is fully-batched and has memory/computational need for cubic power of target length, and quadratic
        power of source length, we need to make sure that each batch has similar length and it does not go over
        max_batch_capacity. We also need to make sure not to include those batches that has less than num_gpu
        sentence pairs (it will crash in multi-gpu).
        MASS refers to https://arxiv.org/pdf/1905.02450.pdf
        """

        paths = glob.glob(batch_pickle_dir + "*")
        self.examples_list = [MassDataset.read_example_file(path) for path in paths]
        print(datetime.datetime.now(), "Done!")

        self.batch_items(max_batch, max_batch_capacity, max_seq_len, src_pad_idx)
        if not keep_examples:
            self.examples_list = []

    def batch_items(self, max_batch, max_batch_capacity, max_seq_len, src_pad_idx):
        print(datetime.datetime.now(), "Building batches")
        self.batches = []
        batches, langs = [], []
        self.lang_ids = set()
        cur_src_batch, cur_langs, cur_max_src_len = [], [], 0
        cur_lex_cand_batch = []
        for examples in self.examples_list:
            for example in examples:
                if len(example[0]) > max_seq_len:
                    continue
                src, lang = example[0], example[1]
                self.lang_ids.add(int(src[0]))
                cur_langs.append(lang)

                cur_max_src_len = max(cur_max_src_len, len(src))

                cur_src_batch.append(src)
                batch_capacity_size = 2 * (cur_max_src_len ** 3) * len(cur_src_batch)
                batch_size = 2 * cur_max_src_len * len(cur_src_batch)

                if (batch_size > max_batch or batch_capacity_size > max_batch_capacity * 1000000) and \
                        len(cur_src_batch[:-1]) >= self.ngpu and len(cur_langs) > 1:
                    batches.append(cur_src_batch[:-1])
                    langs.append(cur_langs[:-1])
                    cur_src_batch = [cur_src_batch[-1]]
                    cur_langs = [cur_langs[-1]]
                    cur_max_src_len = len(cur_src_batch[0])

        if len(cur_src_batch) > 0:
            if len(cur_src_batch) < self.ngpu:
                print("skipping", len(cur_src_batch))
            else:
                batches.append(cur_src_batch)
                langs.append(cur_langs)

        src_pad_idx_find = lambda non_zeros, sz: (sz - 1) if int(non_zeros.size(0)) == 0 else non_zeros[0]
        pad_indices = lambda pads: torch.LongTensor(
            list(map(lambda p: src_pad_idx_find(torch.nonzero(p), int(pads.size(1))), pads)))
        padder = lambda b: pad_sequence(b, batch_first=True, padding_value=src_pad_idx)
        tensorfier = lambda b: list(map(torch.LongTensor, b))
        entry = lambda b, l: {"src_texts": padder(tensorfier(b)), "langs": torch.LongTensor(l)}
        pad_entry = lambda e: {"src_texts": e["src_texts"], "langs": e["langs"],
                               "src_pad_idx": pad_indices(e["src_texts"] == src_pad_idx)}

        self.batches = list(map(lambda b, l: pad_entry(entry(b, l)), batches, langs))

        print("Loaded %d MASS batches!" % (len(self.batches)))
        print("Number of languages", len(self.lang_ids))
        print(datetime.datetime.now(), "Done!")

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        return self.batches[item]


class TextCollator(object):
    def __init__(self, src_pad_idx):
        self.src_pad_idx = src_pad_idx

    def __call__(self, batch):
        langs, batch_text = [], []
        for b in batch:
            batch_text.append(torch.LongTensor(b[0]))
            langs.append(b[1])
        padded_text = pad_sequence(batch_text, batch_first=True, padding_value=self.src_pad_idx)
        pad_mask = (padded_text != self.src_pad_idx)
        return {"texts": padded_text, "pad_mask": pad_mask, "langs": torch.LongTensor(langs)}
