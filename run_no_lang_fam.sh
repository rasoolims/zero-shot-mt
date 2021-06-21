# Create data and train tokenizer
cat /nlp/data/rasooli/wikimatrix-all/all.tok.transliterate.src /scratch-local/bryanli/zsmt/wikimatrix/all.tok.uniq.transliterate.en > /scratch/bryanli/zsmt/wikimatrix/all.tok.transliterate

cd src

python train_tokenizer.py --data /scratch/bryanli/zsmt/wikimatrix/all.tok.transliterate --model /scratch/bryanli/zsmt/tok_best_no_lang_fam --vocab 60000

# Concat Wikimatrix data and split
python scripts/concat_wikimatrix_data.py /nlp/data/rasooli/wikimatrix-all/ /scratch/bryanli/zsmt/wikimatrix/concat

python scripts/icu_transliterate.py /scratch/bryanli/zsmt/wikimatrix/concat.src /scratch/bryanli/zsmt/wikimatrix/concat.srct

NUM_SPLITS=24
mkdir /scratch/bryanli/zsmt/wikimatrix/splits${NUM_SPLITS}/
python scripts/split_data.py /scratch/bryanli/zsmt/wikimatrix/concat.src ${NUM_SPLITS} /scratch/bryanli/zsmt/wikimatrix/splits${NUM_SPLITS}/concat.src.split
python scripts/split_data.py /scratch/bryanli/zsmt/wikimatrix/concat.srct ${NUM_SPLITS} /scratch/bryanli/zsmt/wikimatrix/splits${NUM_SPLITS}/concat.srct.split
python scripts/split_data.py /scratch/bryanli/zsmt/wikimatrix/concat.en ${NUM_SPLITS} /scratch/bryanli/zsmt/wikimatrix/splits${NUM_SPLITS}/concat.en.split

# Create batches for train
python create_mt_batches.py --tok /scratch/bryanli/zsmt/tok_best_no_lang_fam --src /scratch/bryanli/zsmt/wikimatrix/splits/concat.src.split1 --srct /scratch/bryanli/zsmt/wikimatrix/splits/concat.srct.split1 --dst /scratch/bryanli/zsmt/wikimatrix/splits/concat.en.split1 --output /scratch/bryanli/zsmt/output/src2en.train.mt.split1_no_lang_fam

###
# now concat dev
cd /nlp/data/rasooli/wikimatrix-all/eval_data/
cat ar-dev/dev.tok.ar gu-dev/dev.tok.gu kk-dev/dev.tok.kk ro-dev/dev.tok.ro > /scratch/bryanli/zsmt/dev/src
cat ar-dev/dev.tok.en gu-dev/dev.tok.en kk-dev/dev.tok.en ro-dev/dev.tok.en > /scratch/bryanli/zsmt/dev/en
python scripts/icu_transliterate.py /scratch/bryanli/zsmt/dev/src /scratch/bryanli/zsmt/dev/srct

python create_mt_batches.py --tok /scratch/bryanli/zsmt/tok_best --src /scratch/bryanli/zsmt/dev/src --srct /scratch/bryanli/zsmt/dev/srct --dst /scratch/bryanli/zsmt/dev/en --output /scratch/bryanli/zsmt/output/src2en.dev.mt --lang /scratch/bryanli/zsmt/dev/lang_fam.txt

python create_mt_batches.py --tok /scratch/bryanli/zsmt/tok_best_no_lang_fam --src /scratch/bryanli/zsmt/dev/src --srct /scratch/bryanli/zsmt/dev/srct --dst /scratch/bryanli/zsmt/dev/en --output /scratch/bryanli/zsmt/output/src2en.dev.no_lang_fam.mt

# start training
CUDA_VISIBLE_DEVICES=1 python3 -u train_mt.py --tok /scratch/bryanli/zsmt/tok_best_no_lang_fam --model /scratch/bryanli/zsmt/model_no_lang_fam --train /scratch/bryanli/zsmt/output/src2en.train.no_lang_fam.mt.split1 --dev /scratch/bryanli/zsmt/output/src2en.dev.no_lang_fam.mt --batch 15000 --capacity 1200 --step 12500000 --multi --eval-steps 25000|& tee log_no_lang_fam.txt
