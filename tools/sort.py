# encoding: utf-8

import sys
from random import seed as rpyseed
from random import shuffle


# remove_same: reduce same data in the corpus
# shuf: shuffle the data of same source/target length
# max_remove: if one source has several targets, only keep those with highest frequency

def handle(srcfs, srcfm, srcft, tgtfs, tgtfm, tgtft, max_len=256, remove_same=False, shuf=True, max_remove=False):
    def clean(lin):
        rs = []
        for lu in lin:
            if lu:
                rs.append(lu)
        return " ".join(rs), len(rs)

    def filter(ls, lm, lt, max_remove=True):
        tmp = {}
        for us, um, ut in zip(ls, lm, lt):
            if us not in tmp:
                tmp[us] = {(um, ut): 1} if max_remove else set([(um, ut)])
            else:
                if max_remove:
                    tmp[us][(um, ut)] = tmp[us].get((um, ut), 0) + 1
                elif (um, ut) not in tmp[us]:
                    tmp[us].add((um, ut))
        rls, rlm, rlt = [], [], []
        if max_remove:
            for tus, tlt in tmp.items():
                _rs = []
                _maxf = 0
                for key, value in tlt.items():
                    if value > _maxf:
                        _maxf = value
                        _rs = [key]
                    elif value == _maxf:
                        _rs.append(key)
                for tum, tut in _rs:
                    rls.append(tus)
                    rlm.append(tum)
                    rlt.append(tut)
        else:
            for tus, tlt in tmp.items():
                for tum, tut in tlt:
                    rls.append(tus)
                    rlm.append(tum)
                    rlt.append(tut)
        return rls, rlm, rlt

    def shuffle_pair(ls, lm, lt):
        tmp = list(zip(ls, lm, lt))
        shuffle(tmp)
        rs, rm, rt = zip(*tmp)
        return rs, rm, rt

    _max_len = max(1, max_len - 2)

    data = {}

    with open(srcfs, "rb") as fs, open(srcfm, "rb") as fm, open(srcft, "rb") as ft:
        for ls, lm, lt in zip(fs, fm, ft):
            ls, lm, lt = ls.strip(), lm.strip(), lt.strip()
            if ls and lt:
                ls, slen = clean(ls.decode("utf-8").split())
                lm, mlen = clean(lm.decode("utf-8").split())
                lt, tlen = clean(lt.decode("utf-8").split())
                if (slen <= _max_len) and (mlen <= _max_len) and (tlen <= _max_len):
                    slen += mlen
                    lgth = slen + tlen
                    if lgth not in data:
                        data[lgth] = {tlen: [(ls, lm, lt)]}
                    else:
                        if tlen in data[lgth]:
                            data[lgth][tlen].append((ls, lm, lt))
                        else:
                            data[lgth][tlen] = [(ls, lm, lt)]

    length = list(data.keys())
    length.sort()

    ens = "\n".encode("utf-8")

    with open(tgtfs, "wb") as fs, open(tgtfm, "wb") as fm, open(tgtft, "wb") as ft:
        for lgth in length:
            lg = list(data[lgth].keys())
            lg.sort()
            for lu in lg:
                ls, lm, lt = zip(*data[lgth][lu])
                if len(ls) > 1:
                    if remove_same:
                        ls, lm, lt = filter(ls, lm, lt, max_remove)
                    if shuf:
                        ls, lm, lt = shuffle_pair(ls, lm, lt)
                fs.write("\n".join(ls).encode("utf-8"))
                fs.write(ens)
                fm.write("\n".join(lm).encode("utf-8"))
                fm.write(ens)
                ft.write("\n".join(lt).encode("utf-8"))
                ft.write(ens)


if __name__ == "__main__":
    rpyseed(666666)
    handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], int(sys.argv[7]))
