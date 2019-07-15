#!/usr/bin/env bash
export srcd=un-cache
export srctf=src-val.bpe
export srcmf=src-val.mt.bpe
export modelf="expm/debug/checkpoint.t7"
export rsf=trans.txt
export ngpu=1

export cachedir=cache
export dataid=un

export tgtd=$cachedir/$dataid

export bpef=out.bpe

python tools/mktest.py $srcd/$srctf $srcd/$srcmf $tgtd/src.vcb $tgtd/tgt.vcb $tgtd/test.h5 $ngpu
python predict.py $tgtd/$bpef $tgtd/tgt.vcb $modelf
sed -r 's/(@@ )|(@@ ?$)//g' < $tgtd/$bpef > $rsf
