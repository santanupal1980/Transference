# encoding: utf-8

import sys


def handle(srcf, rsf, vsize=32764):
    def clean(lin):
        rs = []
        for lu in lin:
            if lu:
                rs.append(lu)
        return rs

    vocab = {}

    with open(srcf, "rb") as f:
        for line in f:
            tmp = line.strip()
            if tmp:
                for token in clean(tmp.decode("utf-8").split()):
                    vocab[token] = vocab.get(token, 0) + 1

    r_vocab = {}
    for k, v in vocab.items():
        if v not in r_vocab:
            r_vocab[v] = [str(v), k]
        else:
            r_vocab[v].append(k)

    freqs = list(r_vocab.keys())
    freqs.sort(reverse=True)

    ens = "\n".encode("utf-8")
    remain = vsize
    with open(rsf, "wb") as f:
        for freq in freqs:
            cdata = r_vocab[freq]
            ndata = len(cdata) - 1
            if remain < ndata:
                cdata = cdata[:remain + 1]
                ndata = remain
            f.write(" ".join(cdata).encode("utf-8"))
            f.write(ens)
            remain -= ndata
            if remain <= 0:
                break


if __name__ == "__main__":
    handle(sys.argv[1], sys.argv[2], int(sys.argv[3]))
