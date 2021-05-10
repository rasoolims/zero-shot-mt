import os
import tempfile
import unittest

import create_batches
from dataset import TextDataset
from textprocessor import TextProcessor


class TestModel(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName=methodName)

    def test_train_tokenizer(self):
        path_dir_name = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(path_dir_name, "sample.txt")

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor = TextProcessor()
            processor.train_tokenizer([data_path], vocab_size=1000, to_save_dir=tmpdirname, languages={"<en>": 0})
            assert processor.tokenizer.get_vocab_size() == 1000
            sen1 = "Obama signed many landmark bills into law during his first two years in office."
            assert processor._tokenize(sen1) is not None

            many_sens = "\n".join([sen1] * 10)
            assert len(processor.tokenize(many_sens)) == 10

            new_prcoessor = TextProcessor(tok_model_path=tmpdirname)
            assert new_prcoessor.tokenizer.get_vocab_size() == 1000
            sen2 = "Obama signed many landmark bills into law during his first two years in office."
            assert processor._tokenize(sen2) is not None

    def test_data(self):
        path_dir_name = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(path_dir_name, "sample.txt")

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor = TextProcessor()
            processor.train_tokenizer([data_path], vocab_size=1000, to_save_dir=tmpdirname,
                                      languages={"<mzn>": 0, "<glk": 1})
            create_batches.write(text_processor=processor, cache_dir=tmpdirname, seq_len=512, txt_file=data_path,
                                 sen_block_size=10)
            dataset = TextDataset(save_cache_dir=tmpdirname, max_cache_size=3)
            assert dataset.line_num == 70

            dataset.__getitem__(3)
            assert len(dataset.current_cache) == 3

            dataset.__getitem__(9)
            assert len(dataset.current_cache) == 3

            dataset.__getitem__(69)
            assert len(dataset.current_cache) == 2


if __name__ == '__main__':
    unittest.main()
