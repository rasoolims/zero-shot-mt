import datetime
import marshal

from optparse import OptionParser
from typing import Optional
from tqdm import tqdm

from transformers import XLMRobertaTokenizer

from textprocessor import TextProcessor
from utils import get_token_id

def write(tp: TextProcessor, output_file: str, src_txt_file: str, srct_txt_file: str = None,
          dst_txt_file: str = None, shallow: bool = False, lang_lines_path: Optional[str] = None):
    """
    There are scenarios for which the input comes from two streams such as original text and transliterated text. Or we
    want to use two different encoders such as XLM and another one. In these cases, srct_txt_file serves as the second
    file. Note that srct_txt_file is tokenized with the same text_processor that tokenizes the dst_txt_file.
    """
    if not shallow:
        tokenizer_class, weights = XLMRobertaTokenizer, 'xlm-roberta-base'
        tokenizer = tokenizer_class.from_pretrained(weights)
    else:
        tokenizer = tp

    print(datetime.datetime.now(), "Reading source lines!")
    with open(src_txt_file, "r") as s_fp:
        src_lines = list(map(lambda x: x.strip(), s_fp))
    print(datetime.datetime.now(), "Reading target lines!")
    with open(dst_txt_file, "r") as d_fp:
        dst_lines = list(map(lambda x: x.strip(), d_fp))
    print("Number of parallel sentences:", len(dst_lines))

    if lang_lines_path:
        print(datetime.datetime.now(), 'Reading language lines!')
        with open(lang_lines_path, 'r') as l_fp:
            lang_lines = list(map(lambda x: x.strip(), l_fp))
        lang2id = {}
        src_bos_ids = [get_token_id(x, tp, lang2id) for x in lang_lines]
    else:
        print('Not using language lines!')
        bos_id = tp.bos_token_id()
        src_bos_ids = [bos_id] * len(src_lines)

    print(datetime.datetime.now(), "Reading source-translitered lines!")
    if srct_txt_file is None:
        srct_lines = src_lines
    else:
        with open(srct_txt_file, "r") as st_fp:
            srct_lines = list(map(lambda x: x.strip(), st_fp))

    assert len(src_lines) == len(dst_lines) == len(srct_lines)
    if lang_lines_path:
        assert len(lang_lines) == len(src_lines)

    print(datetime.datetime.now(), "Encoding source lines!")
    if shallow:
        src_ids = [encoding.ids for encoding in tqdm(tp.tokenizer.encode_batch(src_lines))]
    else:
        src_ids = tokenizer.batch_encode_plus(tqdm(src_lines)).data['input_ids']

    print(datetime.datetime.now(), "Encoding dest lines!")
    dst_ids = [[tp.bos_token_id()] + encoding.ids + [tp.sep_token_id()] for encoding in
                tqdm(tp.tokenizer.encode_batch(dst_lines))]
    print(datetime.datetime.now(), "Encoding source-translitered lines!")
    srct_ids = [[srct_bos_id] + encoding.ids + [tp.sep_token_id()] for srct_bos_id, encoding in
                zip(src_bos_ids, tqdm(tp.tokenizer.encode_batch(srct_lines)))]
    print(datetime.datetime.now(), "Getting example lengths!")
    example_length = dict(map(lambda e: (e[0], len(e[1])), enumerate(src_ids)))
    print(datetime.datetime.now(), "Sorting example lengths!")
    sorted_lens = sorted(example_length.items(), key=lambda item: item[1])
    print(datetime.datetime.now(), "Getting sorted examples!")
    sorted_examples = list(map(lambda i: (src_ids[i[0]], dst_ids[i[0]], srct_ids[i[0]]), sorted_lens))
    print(datetime.datetime.now(), "Dumping sorted examples!")
    with open(output_file, "wb") as fw:
        marshal.dump(sorted_examples, fw)
    print(datetime.datetime.now(), "Finished!")


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--src", dest="src_data_path", help="Path to the source txt file for xlm tokenizer",
                      metavar="FILE", default=None)
    parser.add_option("--srct", dest="srct_data_path",
                      help="Path to the source txt file for second tokenizer (shallow encoder) ", metavar="FILE",
                      default=None)
    parser.add_option("--dst", dest="dst_data_path", help="Path to the target txt file", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_path", help="Output marshal file ", metavar="FILE", default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--max_seq_len", dest="max_seq_len", help="Max sequence length", type="int", default=175)
    parser.add_option("--min_seq_len", dest="min_seq_len", help="Max sequence length", type="int", default=1)
    parser.add_option("--shallow", action="store_true", dest="shallow_encoder",
                      help="Use shallow encoder instead of XLM", default=False)
    parser.add_option("--lang", dest="lang_lines_path", default='',
            help="path to file with language family IDs for each example in --dst")
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    tokenizer = TextProcessor(options.tokenizer_path)

    write(tp=tokenizer, output_file=options.output_path, src_txt_file=options.src_data_path,
          srct_txt_file=options.srct_data_path, dst_txt_file=options.dst_data_path,
          shallow=options.shallow_encoder, lang_lines_path=options.lang_lines_path)
