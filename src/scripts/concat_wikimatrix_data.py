import os
import sys

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lang_info import get_langs_d

input_dir = os.path.abspath(sys.argv[1])
output_file = os.path.abspath(sys.argv[2])
if len(sys.argv) > 3:
    lang_file = os.path.abspath(sys.argv[3])
else:
    lang_file = ''

with open(output_file + ".en", "w") as enw, open(output_file + ".src", "w") as srcw:
    if lang_file:
        langs_d = get_langs_d(lang_file)
        langf = open(output_file + ".lang_fam.txt", 'w')
    for file in os.listdir(input_dir):
        if file.endswith("tsv"):
            src, tgt = file.split('.')[1].split('-')
            en_src = src == 'en'
            l2 = tgt if en_src else src
            if en_src:
                prefix_len = len("WikiMatrix.en-")
            else:
                prefix_len = len("WikiMatrix.")

            print(file)

            with open(os.path.join(input_dir, file), "r") as r:
                for line in r:
                    spl = line.strip().split("\t")

                    if en_src:
                        en_sen, other_sen = spl[1].strip(), spl[2].strip()
                    else:
                        en_sen, other_sen = spl[2].strip(), spl[1].strip()

                    if len(en_sen) > 1 and len(other_sen) > 1:
                        enw.write(en_sen)
                        enw.write("\n")
                        srcw.write(other_sen)
                        srcw.write("\n")
                        if lang_file:
                            langf.write(langs_d[l2])

    if lang_file:
        langf.close()
print("Finished")
