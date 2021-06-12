# Zero-shot Machine Translation


# Contents
- [Installing Dependencies](#installing-dependencies)
  * [Installation using virtualenv](#installation-using-virtualenv)
  * [Installation using dockers](#installation-using-dockers)
- [Train Machine Translation](#train-machine-translation)
  * [Transliterate the source text!](#transliterate-the-source-text)
  * [Collect raw text for languages:](#collect-raw-text-for-languages)
  * [Train a tokenizer on the concatenation of all raw text:](#train-a-tokenizer-on-the-concatenation-of-all-raw-text)
  * [Train MT on Parallel Data](#train-mt-on-parallel-data)


# Installing Dependencies

## Installation using virtualenv
Here, I use a virtual environment but Conda should be very similar.

1. Create a virtual environment with Python-3
```bash
python3 -m venv [PATH]
```
2. Activate the environment
```
source [PATH]/bin/activate
```

3. Clone the code
```bash
git clone https://github.com/rasoolims/zero-shot-mt
cd zero-shot-mt/src
```

4. Install requirements
In my experiments, I used cuda 10.1 and cudnn 7. To replicate the results, please use the mentioned versions. If things do not work as expected, please use the Docker installation.

```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

It might be the case that ``pyicu`` does not work properly. Follow [its instructions](https://pypi.org/project/PyICU/) to install it.

## Installation using dockers
__Asuuming that Docker and NVIDIA docker is installed.__, follow the following steps:

1. Download the repository and pretrained models
```
git clone https://github.com/rasoolims/zero-shot-mt
```

2. Build the docker in command line:
```bash
docker build dockers/gpu/ -t [docker-name] --no-cache
```

3. Start running the docker:

* Run this with screen since training might take a long time.
```bash
docker run --gpus all -it  [docker-name]
```

## Recreating `lang_info.json` (OPTIONAL)
You may use `lang_info.json` directly, which contains a dictionary with information for each language,
indexed by its ISO 639-1 code. If you wish to re-generate the JSON, you need two dependencies:
```
pip install iso639-lang
git clone git@github.com:cldf-datasets/wals.git
```
Then, you can run
```
python src/lang_info.py -w [path to WALS] -o [path to output JSON name]
```

# Train Machine Translation
Throughout this guideline, I use the small files in the _sample_ folder. Here the Persian and English files are parallel but the Arabic text is not!

__WARNING__: Depending on data, the best parameters might significantly differ. It is good to try some parameter tuning for finding the best setting.


## Transliterate the source text!

```bash
python3 scripts/icu_transliterate.py sample/fa.txt sample/fa.tr.txt
```

## Collect raw text for languages:__

Then, we concatenate the three files. Note that this could be any number of files or languages more than or equal to two.
```bash
cat sample/en.txt sample/fa.tr.txt > sample/all.txt
```


## Train a tokenizer on the concatenation of all raw text:__

Now we are ready to train a tokenizer:
```bash
python train_tokenizer.py --data sample/all.txt --vocab [vocab-size] --model sample/tok
```
The vocab size could be any value. Anything between 30000 to 100000 should be good. For this sample file, try 1000.

### Add language family information to the tokenizer (OPTIONAL)
You can initialize the tokenizer with language families by adding `--lang`:
```bash
python train_tokenizer.py --data sample/all.txt --vocab [vocab-size] --model sample/tok --lang lang_info.json
```


## Train MT on Parallel Data
Parallel data could be gold-standard or mined. You should load pre-trained MASS models for the best performance.

__1. Create binary files for training and dev dataset:__
For simplicity, we use the Persian and English text files as both training and development datasets by using their last 100 sentences as development data.
```bash
head -9900 sample/fa.txt > sample/train.fa
head -9900 sample/en.txt > sample/train.en
tail -100 sample/en.txt > sample/dev.en
tail -100 sample/fa.txt > sample/dev.fa
head -9900 sample/fa.tr.txt > sample/train.tr.fa
tail -100 sample/fa.tr.txt > sample/dev.tr.fa

python create_mt_batches.py --tok sample/tok/ --src sample/train.fa \
 --dst sample/train.en --srct sample/train.tr.fa\
 --output sample/fa2en.train.mt

python create_mt_batches.py --tok sample/tok/ --src sample/dev.fa \
 --dst sample/train.en --srct sample/dev.tr.fa  \
 --output sample/fa2en.dev.mt

```

If you create translation data in multiple direction, you can train multilingual translation for which we learn translation from multiple directions. Multiple data files can be separated by ``,`` in the arguments both for ``--train_mt`` and ``--dev_mt`` options.

__2. Train machine translation:__
```
 CUDA_VISIBLE_DEVICES=0 python3 -u train_mt.py --tok  sample/tok/ \
 --model sample/mt_model  --train_mt sample/fa2en.train.mt \
 --capacity 600 --batch 4000   --beam 4 --step 500000 --warmup 4000 \
 --lr 0.0001  --dev_mt sample/fa2en.dev.mt \
 --dropout 0.1 --fp16 --multi
```

After you are done, you can use the model path ``sample/mt_model`` for translating text to English (similar to [the section on using the pretrained models in our paper](#translation).

__3. Translate:__
```bash
CUDA_VISIBLE_DEVICES=0 python -u translate.py --tok sample/tok/ \
--model sample/mt_model --input sample/dev.fa --input2 sample/dev.tr.fa\
--output sample/dev.output.en
```
Note that there is a ``--verbose`` option where it puts the input and output lines separated by ``|||``. This is useful especially if you want to use it for back-translation (to make sure that sentence alignments are completely guaranteed), or for annotation projection in which you might need it for word alignment.
