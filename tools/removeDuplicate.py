import sys


def handle(src, trg, src_out, trg_out):
    lines_seen = set()  # holds lines already seen
    outfile_src = open(src_out, "w", encoding="utf8")
    infile_src = open(src, "r", encoding="utf8")

    outfile_trg = open(trg_out, "w", encoding="utf8")
    infile_trg = open(trg, "r", encoding="utf8")

    print("Removing duplicates")
    for line1, line2 in zip(infile_src, infile_trg):

        if line1 not in lines_seen or line2 not in lines_seen:  # not a duplicate
            outfile_src.write(line1)
            outfile_trg.write(line2)
            lines_seen.add(line1)
            lines_seen.add(line2)
    outfile_src.close()
    outfile_trg.close()


if __name__ == "__main__":
    handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
