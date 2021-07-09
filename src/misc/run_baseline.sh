# Concat Wikimatrix data and split
python scripts/concat_wikimatrix_data.py /nlp/data/rasooli/wikimatrix-all/ /scratch/bryanli/zsmt/wikimatrix/concat

cat /scratch/bryanli/zsmt/wikimatrix/concat.en /scratch/bryanli/zsmt/wikimatrix/concat.src > /scratch/bryanli/zsmt/wikimatrix/concat.all

# train tokenizer
# we use 100k since this is not transliterated.
python train_tokenizer.py --data /scratch/bryanli/zsmt/wikimatrix/concat.all --model /scratch/bryanli/zsmt/tok_baseline --vocab 100000

NUM_SPLITS=24
mkdir /scratch/bryanli/zsmt/wikimatrix/splits${NUM_SPLITS}/
python scripts/split_data.py /scratch/bryanli/zsmt/wikimatrix/concat.src ${NUM_SPLITS} /scratch/bryanli/zsmt/wikimatrix/splits${NUM_SPLITS}/concat.src.split
python scripts/split_data.py /scratch/bryanli/zsmt/wikimatrix/concat.en ${NUM_SPLITS} /scratch/bryanli/zsmt/wikimatrix/splits${NUM_SPLITS}/concat.en.split

# Create batches for train
python create_mt_batches.py --tok /scratch/bryanli/zsmt/tok_baseline --src /scratch/bryanli/zsmt/wikimatrix/splits/concat.src.split1 --dst /scratch/bryanli/zsmt/wikimatrix/splits/concat.en.split1 --output /scratch/bryanli/zsmt/output/src2en.train.baseline.mt.split1

###
# now concat dev
cd /nlp/data/rasooli/wikimatrix-all/eval_data/
cat ar-dev/dev.tok.ar gu-dev/dev.tok.gu kk-dev/dev.tok.kk ro-dev/dev.tok.ro > /scratch/bryanli/zsmt/dev/src
cat ar-dev/dev.tok.en gu-dev/dev.tok.en kk-dev/dev.tok.en ro-dev/dev.tok.en > /scratch/bryanli/zsmt/dev/en

python create_mt_batches.py --tok /scratch/bryanli/zsmt/tok_baseline --src /scratch/bryanli/zsmt/dev/src --dst /scratch/bryanli/zsmt/dev/en --output /scratch/bryanli/zsmt/output/src2en.dev.baseline.mt

# start training
CUDA_VISIBLE_DEVICES=1 python3 -u train_mt.py --tok /scratch/bryanli/zsmt/tok_baseline --model /scratch/bryanli/zsmt/model_baseline --train /scratch/bryanli/zsmt/output/src2en.train.baseline.mt.split1 --dev /scratch/bryanli/zsmt/output/src2en.dev.baseline.mt --batch 15000 --capacity 1200 --step 12500000 --eval-steps 25000|& tee log_baseline.txt
