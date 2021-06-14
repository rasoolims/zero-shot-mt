import json
import os
from optparse import OptionParser
from typing import Optional

from textprocessor import TextProcessor
from lang_info import get_langs_d

def get_tokenizer(train_path: Optional[str] = None,
                  model_path: Optional[str] = None, vocab_size: Optional[int] = None,
                  lang_path: Optional[str] = None) -> TextProcessor:
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    langs = {}
    if lang_path:
        print('Loading languages dict...')
        langs = get_langs_d(lang_path)
    else:
        print('Not using languages dict')

    print("Training Tokenizer...")
    text_processor = TextProcessor()
    text_processor.train_tokenizer(
        paths=[train_path], vocab_size=vocab_size, to_save_dir=model_path, languages=langs)
    print("done!")


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--data", dest="data_path", help="Path to the data folder",
                      metavar="FILE", default=None)
    parser.add_option("--vocab", dest="vocab_size",
                      help="Vocabulary size", type="int", default=30000)
    parser.add_option("--model", dest="model_path", help="Directory path to save the best model", metavar="FILE",
                      default=None)
    parser.add_option("--lang", dest="lang_path", help="path to language info file", default='')
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    tokenizer = get_tokenizer(train_path=options.data_path,
                              model_path=options.model_path, vocab_size=options.vocab_size,
                              lang_path=options.lang_path)
