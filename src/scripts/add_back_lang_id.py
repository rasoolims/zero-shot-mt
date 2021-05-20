import os
import sys

comment = """
Having the first argument as id-augmented file, the second argument as tokenized raw file, this scripts writes the tokenized
files with id-agumented information.

E.g.
First file line: "<en> This is a sentence. </s>"
Second file line:  "This is a sentence ."
Output line: "<en> This is a sentence . </s>"
"""

if len(sys.argv) < 3:
    print(comment)
    sys.exit(0)

with open(os.path.abspath(sys.argv[1]), "r") as idr, open(os.path.abspath(sys.argv[2]), "r") as tokr, open(
        os.path.abspath(sys.argv[3]), "w") as w:
    for idl, tokl in zip(idr, tokr):
        spl = idl.strip().split(" ")
        line = " ".join([spl[0], tokl.strip(), spl[-1]])
        w.write(line)
        w.write("\n")
