# Create data and train tokenizer
cat /nlp/data/rasooli/wikimatrix-all/all.tok.transliterate.src /scratch-local/bryanli/zsmt/wikimatrix/all.tok.uniq.transliterate.en > /scratch/bryanli/zsmt/wikimatrix/all.tok.transliterate
ll /scratch/bryanli/zsmt/wikimatrix/all.tok.transliterate /nlp/data/rasooli/wikimatrix-all/all.tok

cd src

python lang_info.py

python train_tokenizer.py --data /scratch/bryanli/zsmt/wikimatrix/all.tok.transliterate --model /scratch/bryanli/zsmt/tok_best --lang ../lang_info.json --vocab 60000

# Concat Wikimatrix data and split
python scripts/concat_wikimatrix_data.py /nlp/data/rasooli/wikimatrix-all/ /scratch/bryanli/zsmt/wikimatrix/concat ../lang_info.json

python scripts/icu_transliterate.py /scratch/bryanli/zsmt/wikimatrix/concat.src /scratch/bryanli/zsmt/wikimatrix/concat.srct

NUM_SPLITS=24
mkdir /scratch/bryanli/zsmt/wikimatrix/splits${NUM_SPLITS}/
python scripts/split_data.py /scratch/bryanli/zsmt/wikimatrix/concat.src ${NUM_SPLITS} /scratch/bryanli/zsmt/wikimatrix/splits${NUM_SPLITS}/concat.src.split
python scripts/split_data.py /scratch/bryanli/zsmt/wikimatrix/concat.srct ${NUM_SPLITS} /scratch/bryanli/zsmt/wikimatrix/splits${NUM_SPLITS}/concat.srct.split
python scripts/split_data.py /scratch/bryanli/zsmt/wikimatrix/concat.en ${NUM_SPLITS} /scratch/bryanli/zsmt/wikimatrix/splits${NUM_SPLITS}/concat.en.split
python scripts/split_data.py /scratch/bryanli/zsmt/wikimatrix/concat.lang_fam.txt ${NUM_SPLITS} /scratch/bryanli/zsmt/wikimatrix/splits${NUM_SPLITS}/concat.lang_fam.txt.split

# Create batches for train
python create_mt_batches.py --tok /scratch/bryanli/zsmt/tok_best --src /scratch/bryanli/zsmt/wikimatrix/splits/concat.src.split1 --srct /scratch/bryanli/zsmt/wikimatrix/splits/concat.srct.split1 --dst /scratch/bryanli/zsmt/wikimatrix/splits/concat.en.split1 --output /scratch/bryanli/zsmt/output/src2en.train.mt.split1 --lang /scratch/bryanli/zsmt/wikimatrix/splits/concat.lang_fam.txt.split1

###
# now concat dev
cd /nlp/data/rasooli/wikimatrix-all/eval_data/
cat ar-dev/dev.tok.ar gu-dev/dev.tok.gu kk-dev/dev.tok.kk ro-dev/dev.tok.ro > /scratch/bryanli/zsmt/dev/src
cat ar-dev/dev.tok.en gu-dev/dev.tok.en kk-dev/dev.tok.en ro-dev/dev.tok.en > /scratch/bryanli/zsmt/dev/en
python scripts/icu_transliterate.py /scratch/bryanli/zsmt/dev/src /scratch/bryanli/zsmt/dev/srct

# create lang families txt
AR_CNT=$(wc -l < ar-dev/dev.tok.ar)
GU_CNT=$(wc -l < gu-dev/dev.tok.gu)
KK_CNT=$(wc -l < kk-dev/dev.tok.kk)
RO_CNT=$(wc -l < ro-dev/dev.tok.ro)
cat <(yes "<Semitic>" | head -n $AR_CNT) <(yes "<Indic>" | head -n $GU_CNT) <(yes "<Turkic>" | head -n $KK_CNT) <(yes "<Romance>" | head -n $RO_CNT) > /scratch/bryanli/zsmt/dev/lang_fam.txt

python create_mt_batches.py --tok /scratch/bryanli/zsmt/tok_best --src /scratch/bryanli/zsmt/dev/src --srct /scratch/bryanli/zsmt/dev/srct --dst /scratch/bryanli/zsmt/dev/en --output /scratch/bryanli/zsmt/output/src2en.dev.mt --lang /scratch/bryanli/zsmt/dev/lang_fam.txt

# start training
CUDA_VISIBLE_DEVICES=0 python -u train_mt.py --tok /scratch/bryanli/zsmt/tok_best --model /scratch/bryanli/zsmt/model_lang_fam --train /scratch/bryanli/zsmt/output/src2en.train.mt.split1 --dev /scratch/bryanli/zsmt/output/src2en.dev.mt --batch 20000 --capacity 1000 --step 12500000 --fp16 --multi &| tee log.txt

CUDA_VISIBLE_DEVICES=0 python3 -u train_mt.py --tok /scratch/bryanli/zsmt/tok_best --model /scratch/bryanli/zsmt/model_lang_fam --train /scratch/bryanli/zsmt/output/src2en.train.mt.split1 --dev /scratch/bryanli/zsmt/output/src2en.dev.mt --batch 15000 --capacity 1200 --step 12500000 --multi --eval-steps 25000
