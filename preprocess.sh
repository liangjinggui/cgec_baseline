DATA='data' # input dir
OUT='out' # output dir

DATA_BIN=$OUT/data_bin
DATA_RAW=$OUT/data_raw
mkdir -p $DATA_BIN
mkdir -p $DATA_RAW
# 清空已缓存的数据
rm -rf $DATA_BIN
rm -rf $DATA_RAW

trainpref='data/train'
validpref='data/valid'
python preprocess.py \
--source-lang source \
--target-lang target \
--trainpref $trainpref \
--validpref $validpref \
--destdir $DATA_BIN \
--padding-factor 1 \
--joined-dictionary \
--copy-ext-dict \
--output-format binary \

# preprocess test
python preprocess.py \
--source-lang source \
--target-lang target \
--only-source \
--destdir $DATA_RAW \
--padding-factor 1 \
--joined-dictionary \
--srcdict $DATA_BIN/dict.source.txt \
--copy-ext-dict \
--testpref data/test \
--output-format raw
