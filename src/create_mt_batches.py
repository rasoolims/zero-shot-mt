import datetime
import marshal
from optparse import OptionParser

from transformers import XLMRobertaTokenizer

from textprocessor import TextProcessor


def write(tp: TextProcessor, output_file: str, src_txt_file: str, src_lang: int, srct_txt_file: str = None,
          dst_txt_file: str = None, dst_lang: int = None, min_len: int = 1, max_len: int = 175, shallow: bool = False):
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

    print(datetime.datetime.now(), "Reading source-translitered lines!")
    if srct_txt_file is None:
        srct_lines = src_lines
    else:
        with open(srct_txt_file, "r") as st_fp:
            srct_lines = list(map(lambda x: x.strip(), st_fp))

    assert len(src_lines) == len(dst_lines) == len(srct_lines)
    print(datetime.datetime.now(), "Number of parallel sentences:", len(dst_lines))
    src_encode = lambda x: tp.tokenize_one_sentence_with_langid(x, src_lang) if shallow else tokenizer.encode(x)
    target_encode = lambda x: tp.tokenize_one_sentence_with_langid(x, dst_lang)
    srct_enclode = lambda x: tp.tokenize_one_sentence(x)

    print(datetime.datetime.now(), "Tokenizing examples!")
    examples = list(map(lambda x: (src_encode(x[0]), target_encode(x[1]), srct_enclode(x[2])),
                        zip(src_lines, dst_lines, dst_lines)))
    print(datetime.datetime.now(), "Getting example lengths!")
    example_length = dict(map(lambda e: (e[0], len(e[1][0])), enumerate(examples)))
    print(datetime.datetime.now(), "Sorting example lengths!")
    sorted_lens = sorted(example_length.items(), key=lambda item: item[1])
    print(datetime.datetime.now(), "Getting sorted examples!")
    sorted_examples = list(map(lambda i: examples[i[0]], sorted_lens))
    print(datetime.datetime.now(), "Dumping sorted examples!")
    with open(output_file, "wb") as fw:
        dst_lang_id = tp.languages[tp.id2token(dst_lang)]
        marshal.dump((sorted_examples, dst_lang_id), fw)
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
    parser.add_option("--src-lang", dest="src_lang", type="str", help="Only use it with the --shallow option",
                      default=None)
    parser.add_option("--dst-lang", dest="dst_lang", type="str", default=None)
    parser.add_option("--shallow", action="store_true", dest="shallow_encoder",
                      help="Use shallow encoder instead of XLM", default=False)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    tokenizer = TextProcessor(options.tokenizer_path)

    src_lang = None if options.src_lang is None else tokenizer.token_id("<" + options.src_lang + ">")
    dst_lang = tokenizer.token_id("<" + options.dst_lang + ">") if options.dst_lang is not None else None
    write(tp=tokenizer, output_file=options.output_path, src_txt_file=options.src_data_path,
          srct_txt_file=options.srct_data_path, src_lang=src_lang,
          dst_txt_file=options.dst_data_path, dst_lang=dst_lang, shallow=options.shallow_encoder)
