import logging
import marshal

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MTDataset(Dataset):
    def __init__(self, max_batch_capacity: int, max_batch: int,
                 src_pad_idx: int, dst_pad_idx: int, max_seq_len: int = 175, batch_pickle_dir: str = None,
                 keep_src_pad_idx=True, ngpu=1, examples=None):
        self.keep_src_pad_idx = keep_src_pad_idx
        self.ngpu = ngpu

        if examples is None:
            self.build_batches(batch_pickle_dir, max_batch_capacity, max_batch, src_pad_idx, dst_pad_idx, max_seq_len)
        else:
            self.build_batches_from_examples(dst_pad_idx, examples, max_batch, max_batch_capacity, max_seq_len,
                                             src_pad_idx)

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
            examples = marshal.load(fr)

            self.build_batches_from_examples(dst_pad_idx, examples, max_batch, max_batch_capacity, max_seq_len,
                                             src_pad_idx)

            print("\nLoaded %d bitext sentences to %d batches!" % (len(examples), len(self.batches)))

    def build_batches_from_examples(self, dst_pad_idx, examples, max_batch, max_batch_capacity, max_seq_len,
                                    src_pad_idx):
        print("BUILDING MT BATCHES")
        self.batches = []
        cur_src_batch, cur_srct_batch, cur_dst_batch = [], [], []
        cur_max_src_len, cur_max_srct_len, cur_max_dst_len = 0, 0, 0
        for ei, example in enumerate(examples):
            src = torch.LongTensor(example[0][:max_seq_len])  # trim if longer than expected!
            dst = torch.LongTensor(example[1][:max_seq_len])  # trim if longer than expected!
            srct = torch.LongTensor(example[2][:max_seq_len])  # trim if longer than expected!
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
                    len(cur_src_batch[:-1]) >= self.ngpu and len(cur_src_batch) > 1:
                src_batch = pad_sequence(cur_src_batch[:-1], batch_first=True, padding_value=src_pad_idx)
                # Bellow should be dst_pad_idx since srct has the same tokenizer as the target output.
                srct_batch = pad_sequence(cur_srct_batch[:-1], batch_first=True, padding_value=dst_pad_idx)
                dst_batch = pad_sequence(cur_dst_batch[:-1], batch_first=True, padding_value=dst_pad_idx)
                src_pad_mask = (src_batch != src_pad_idx)
                # Bellow should be dst_pad_idx since srct has the same tokenizer as the target output.
                srct_pad_mask = (srct_batch != dst_pad_idx)
                dst_pad_mask = (dst_batch != dst_pad_idx)

                entry = {"src_texts": src_batch, "srct_texts": srct_batch, "src_pad_mask": src_pad_mask,
                         "dst_texts": dst_batch, "srct_pad_mask": srct_pad_mask, "dst_pad_mask": dst_pad_mask}
                self.batches.append(entry)
                cur_src_batch, cur_srct_batch = [cur_src_batch[-1]], [cur_srct_batch[-1]]
                cur_dst_batch = [cur_dst_batch[-1]]
                cur_max_src_len, cur_max_srct_len, cur_max_dst_len = int(cur_src_batch[0].size(0)), int(
                    cur_srct_batch[0].size(0)), int(cur_dst_batch[0].size(0))

            if ei % 1000 == 0:
                print(ei, "/", len(examples), end="\r")
        if len(cur_src_batch) > 0 and len(cur_src_batch) >= self.ngpu:
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
                     "dst_pad_mask": dst_pad_mask}
            self.batches.append(entry)
        if self.keep_src_pad_idx:
            src_pad_idx_find = lambda non_zeros, sz: (sz - 1) if int(non_zeros.size(0)) == 0 else non_zeros[0]
            for b in self.batches:
                pads = b["src_texts"] == src_pad_idx
                sz = int(pads.size(1))
                pad_indices = torch.LongTensor(list(map(lambda p: src_pad_idx_find(torch.nonzero(p), sz), pads)))
                b["src_pad_idx"] = pad_indices

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        return self.batches[item]
