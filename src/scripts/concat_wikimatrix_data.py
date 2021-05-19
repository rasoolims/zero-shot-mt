import os
import sys

input_dir = os.path.abspath(sys.argv[1])
output_file = os.path.abspath(sys.argv[2])
with open(output_file + ".en", "w") as enw, open(output_file + ".src", "w") as srcw:
    for file in os.listdir(input_dir):
        if file.endswith("tsv"):
            en_src = "en-" in file  # English is source!
            if en_src:
                prefix_len = len("WikiMatrix.en-")
                dst_lang_id = "<" + file[prefix_len:file.rfind(".")] + ">"
            else:
                prefix_len = len("WikiMatrix.")
                dst_lang_id = "<" + file[prefix_len:file.rfind("-")] + ">"

            print(file, dst_lang_id)
            with open(os.path.join(input_dir, file), "r") as r:
                for line in r:
                    spl = line.strip().split("\t")

                    if en_src:
                        en_sen, other_sen = spl[1].strip(), spl[2].strip()
                    else:
                        en_sen, other_sen = spl[2].strip(), spl[1].strip()

                    if len(en_sen) > 1 and len(other_sen) > 1:
                        en_sen = " ".join(["<en>", en_sen, "</s>"])
                        other_sen = " ".join([dst_lang_id, other_sen, "</s>"])
                        enw.write(en_sen)
                        enw.write("\n")
                        srcw.write(other_sen)
                        srcw.write("\n")

print("Finished")
