import os
import pickle
from typing import Dict, List, Optional

from tokenizers import Encoding
from tokenizers import SentencePieceBPETokenizer
from tokenizers.normalizers import BertNormalizer


class TextProcessor:
    def __init__(self, tok_model_path: Optional[str] = None):
        self.languages = {}
        if tok_model_path is not None:
            self.tokenizer = SentencePieceBPETokenizer(
                tok_model_path + "/vocab.json",
                tok_model_path + "/merges.txt",
            )
            with open(os.path.join(tok_model_path, "langs"), "rb") as fp:
                self.languages: Dict[str, int] = pickle.load(fp)
        self.init_properties(self.languages)

    def init_properties(self, languages: Dict[str, int] = None):
        self.max_len = 512
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        self.unk_token = "<unk>"
        self.sep_token = "</s>"
        self.bos = "<s>"
        self.special_tokens = [self.pad_token, self.bos, self.unk_token, self.mask_token,
                               self.sep_token] + list(languages.keys())
        self.languages = languages

    def train_tokenizer(self, paths: List[str], vocab_size: int, to_save_dir: str, languages: Dict[str, int]):
        self.tokenizer = SentencePieceBPETokenizer()
        bert_normalizer = BertNormalizer(clean_text=True, handle_chinese_chars=False, lowercase=False)
        self.tokenizer._tokenizer.normalizer = bert_normalizer
        self.init_properties(languages)
        self.tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=5, special_tokens=self.special_tokens)
        self.save(directory=to_save_dir)

    def _tokenize(self, line) -> Encoding:
        return self.tokenizer.encode(line)

    def save(self, directory):
        self.tokenizer.save_model(directory)
        with open(os.path.join(directory, "langs"), "wb") as fp:
            pickle.dump(self.languages, fp)

    def pad_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.pad_token)

    def mask_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.mask_token)

    def unk_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.unk_token)

    def bos_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.bos)

    def sep_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.sep_token)

    def token_id(self, token: str) -> int:
        tok_id = self.tokenizer.token_to_id(token)
        if tok_id is None:
            return 0
        return tok_id

    def id2token(self, id: int) -> str:
        return self.tokenizer.id_to_token(id)

    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
