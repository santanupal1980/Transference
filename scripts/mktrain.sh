#!/usr/bin/env bash
#!/bin/bash

# take the processed data from scripts/mkbpe.sh and convert to tensor representation.

export cachedir=cache
export dataid=ape_data

export working_dir=$cachedir/$dataid

export vsize=32000
export maxtokens=256
export ngpu=1

export src_tf=all.bpe.src
export mt_tf=all.bpe.mt
export pe_tf=all.bpe.pe
export src_vf=dev.bpe.src
export mt_vf=dev.bpe.mt
export pe_vf=dev.bpe.pe

python tools/sort.py $working_dir/$src_tf $working_dir/$mt_tf $working_dir/$pe_tf $working_dir/src.train.srt $working_dir/mt.train.srt $working_dir/tgt.train.srt $maxtokens
python tools/sort.py $working_dir/$src_vf $working_dir/$mt_vf $working_dir/$pe_vf $working_dir/src.dev.srt $working_dir/mt.dev.srt $working_dir/tgt.dev.srt 1048576

python tools/vocab.py $working_dir/src.train.srt $working_dir/src.vcb $vsize
python tools/share_vocab.py $working_dir/tgt.train.srt  $working_dir/mt.train.srt $working_dir/tgt.vcb $vsize
# use the following line if you want a shared vocabulary
#python tools/share_vocab.py $working_dir/src.train.srt $working_dir/tgt.train.srt $working_dir/common.vcb $vsize

python tools/mkiodata.py $working_dir/src.train.srt $working_dir/mt.train.srt $working_dir/tgt.train.srt $working_dir/src.vcb $working_dir/tgt.vcb $working_dir/train.h5 $ngpu
python tools/mkiodata.py $working_dir/src.dev.srt $working_dir/mt.dev.srt $working_dir/tgt.dev.srt $working_dir/src.vcb $working_dir/tgt.vcb $working_dir/dev.h5 $ngpu