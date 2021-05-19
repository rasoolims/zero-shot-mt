import datetime
import marshal
from optparse import OptionParser

from transformers import XLMRobertaTokenizer

from textprocessor import TextProcessor


def write(text_processor: TextProcessor, output_file: str, src_txt_file: str, src_lang: int, srct_txt_file: str = None,
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
        tokenizer = text_processor

    examples = {}
    line_num = 0
    src_lang_str = text_processor.languages[text_processor.id2token(src_lang)]
    lens = {}
    if srct_txt_file is not None:
        dst_lang_str = text_processor.languages[text_processor.id2token(dst_lang)]
        with open(src_txt_file, "r") as s_fp, open(dst_txt_file, "r") as d_fp, open(srct_txt_file, "r") as st_fp:
            for src_line, dst_line, srct_line in zip(s_fp, d_fp, st_fp):
                if len(src_line.strip()) == 0 or len(dst_line.strip()) == 0: continue
                if not shallow:
                    src_tok_line = tokenizer.encode(src_line.strip())
                else:
                    src_tok_line = text_processor.tokenize_one_sentence_with_langid(src_line.strip(), src_lang)

                srct_tok_line = text_processor.tokenize_one_sentence(srct_line.strip())
                dst_tok_line = text_processor.tokenize_one_sentence_with_langid(dst_line.strip(), dst_lang)

                if min_len <= len(src_tok_line) <= max_len and min_len <= len(
                        dst_tok_line) <= max_len and min_len <= len(srct_tok_line) <= max_len:
                    examples[line_num] = (src_tok_line, dst_tok_line, dst_lang_str, srct_tok_line)
                    lens[line_num] = len(dst_tok_line)
                    line_num += 1

                if line_num % 1000 == 0:
                    print(line_num, end="\r")

        print("\nSorting")
        sorted_lens = sorted(lens.items(), key=lambda item: item[1])
        sorted_examples = []
        print("Sorted examples")
        for len_item in sorted_lens:
            line_num = len(sorted_examples)
            sorted_examples.append(examples[len_item[0]])

        print("Dumping")
        with open(output_file, "wb") as fw:
            marshal.dump(sorted_examples, fw)
    else:
        dst_lang_str = text_processor.languages[text_processor.id2token(dst_lang)]
        with open(src_txt_file, "r") as s_fp, open(dst_txt_file, "r") as d_fp:
            for src_line, dst_line in zip(s_fp, d_fp):
                if len(src_line.strip()) == 0 or len(dst_line.strip()) == 0: continue
                if not shallow:
                    src_tok_line = tokenizer.encode(src_line.strip())
                else:
                    src_tok_line = text_processor.tokenize_one_sentence_with_langid(src_line.strip(), src_lang)

                srct_tok_line = text_processor.tokenize_one_sentence(src_line.strip())
                dst_tok_line = text_processor.tokenize_one_sentence_with_langid(dst_line.strip(), dst_lang)

                if min_len <= len(src_tok_line) <= max_len and min_len <= len(dst_tok_line) <= max_len:
                    examples[line_num] = (src_tok_line, dst_tok_line, dst_lang_str, srct_tok_line)
                    lens[line_num] = len(dst_tok_line)
                    line_num += 1

                if line_num % 1000 == 0:
                    print(line_num, end="\r")

        print("\nSorting")
        sorted_lens = sorted(lens.items(), key=lambda item: item[1])
        sorted_examples = []
        print("Sorted examples")
        for len_item in sorted_lens:
            line_num = len(sorted_examples)
            sorted_examples.append(examples[len_item[0]])

        print("Dumping")
        with open(output_file, "wb") as fw:
            marshal.dump(sorted_examples, fw)


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

    print(datetime.datetime.now(), "Writing batches")
    src_lang = tokenizer.token_id("<" + options.src_lang + ">")
    dst_lang = tokenizer.token_id("<" + options.dst_lang + ">") if options.dst_lang is not None else None
    write(text_processor=tokenizer, output_file=options.output_path, src_txt_file=options.src_data_path,
          srct_txt_file=options.srct_data_path,
          dst_txt_file=options.dst_data_path, dst_lang=dst_lang, shallow=options.shallow_encoder)
    print(datetime.datetime.now(), "Finished")
