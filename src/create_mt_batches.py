import datetime
import marshal

from optparse import OptionParser
from typing import Optional

from transformers import XLMRobertaTokenizer

from textprocessor import TextProcessor
from lang_info import get_langs_d


def write(tp: TextProcessor, output_file: str, src_txt_file: str, srct_txt_file: str = None,
          dst_txt_file: str = None, shallow: bool = False, lang_path: Optional[str] = None):
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
    src = src_txt_file.rsplit('.', 1)[-1]
    dst = dst_txt_file.rsplit('.', 1)[-1]

    langs = {}
    if lang_path:
        print('Loading languages dict...')
        langs = get_langs_d(lang_path)
    else:
        print('Not using languages dict')

    print(datetime.datetime.now(), "Reading source lines!")
    with open(src_txt_file, "r") as s_fp:
        src_lines = list(map(lambda x: x.strip(), s_fp))
    print(datetime.datetime.now(), "Reading target lines!")
    with open(dst_txt_file, "r") as d_fp:
        dst_lines = list(map(lambda x: x.strip(), d_fp))

    print(datetime.datetime.now(), "Reading source-translitered lines!")
    if srct_txt_file is None:
        srct_lines = src_lines
    else:
        with open(srct_txt_file, "r") as st_fp:
            srct_lines = list(map(lambda x: x.strip(), st_fp))

    assert len(src_lines) == len(dst_lines) == len(srct_lines)
    print(datetime.datetime.now(), "Number of parallel sentences:", len(dst_lines))
    if shallow:
        src_ids = [encoding.ids for encoding in tp.tokenizer.encode_batch(src_lines)]
    else:
        src_ids = tokenizer.batch_encode_plus(src_lines).data['input_ids']
    dst_bos_token_id = tp.bos_token_id() if not langs else tp.token_id(langs[dst])
    srct_bos_token_id = tp.bos_token_id() if not langs else tp.token_id(langs[src])
    dst_ids = [[dst_bos_token_id] + encoding.ids + [tp.sep_token_id()] for encoding in
               tp.tokenizer.encode_batch(dst_lines)]
    srct_ids = [[srct_bos_token_id] + encoding.ids + [tp.sep_token_id()] for encoding in
                tp.tokenizer.encode_batch(srct_lines)]

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
    parser.add_option("--lang", dest="lang_path", help="path to language info file", default='')
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    tokenizer = TextProcessor(options.tokenizer_path)

    write(tp=tokenizer, output_file=options.output_path, src_txt_file=options.src_data_path,
          srct_txt_file=options.srct_data_path, dst_txt_file=options.dst_data_path,
          shallow=options.shallow_encoder, lang_path=options.lang_path)
