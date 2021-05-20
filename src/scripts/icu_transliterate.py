import os
import sys

import icu

tl = icu.Transliterator.createInstance('Any-Latin; Latin-ASCII')

with open(os.path.abspath(sys.argv[1]), "r") as r, open(os.path.abspath(sys.argv[2]), "w") as w:
    for i, line in enumerate(r):
        transliteration = tl.transliterate(line.strip())
        w.write(transliteration)
        w.write("\n")
        print(i, end="\r")
print("\n Finished!")
