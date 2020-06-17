import os
import sys


def extract_doctences(line, min_len):
    line = line.strip()
    if len(line) == 0:
        return []

    docs = line.split("</s>")
    doc_split = docs[0].strip().split(" ")
    docs[0] = " ".join(doc_split[1:])
    lang = doc_split[0]
    len_condition = lambda s: len(s.strip()) > 0 and len(s.strip().split(" ")) > min_len
    return list(filter(lambda x: x is not None,
                       map(lambda s: " ".join([lang, s.strip(), "</s>"]) if len_condition(s) else None, docs)))


path = os.path.abspath(sys.argv[1])
output_path = os.path.abspath(sys.argv[2])

with open(path, "r") as reader, open(output_path, "w") as writer:
    for i, line in enumerate(reader):
        sens = extract_doctences(line, 0)
        if len(sens) > 0:
            writer.write("\n".join(sens))
            writer.write("\n")
        print(i, end="\r")
print("Done!")
