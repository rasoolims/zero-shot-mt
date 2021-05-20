import datetime
import os
import sys

print(datetime.datetime.now(), "Reading lines!")
with open(os.path.abspath(sys.argv[1]), "r") as s_fp:
    lines = list(map(lambda x: x.strip(), s_fp))

num_splits = int(sys.argv[2])

for s in range(num_splits):
    print(datetime.datetime.now(), "Write split", str(s + 1))
    with open(os.path.abspath(sys.argv[3]) + str(s + 1), "w") as w:
        for i in range(s, len(lines), step=num_splits):
            w.write(lines[i].strip() + "\n")
print(datetime.datetime.now(), "Finished!")
