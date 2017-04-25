import os
import sys

every = int(sys.argv[1])

check = {}
check_list = []
for filename in os.listdir("."):
        if filename.endswith("meta"):
                checkpoint = int(filename.split("-")[1].split(".")[0])
                check_list.append(checkpoint)

check_list.sort()
for checkpoint in check_list:
        if check.has_key(checkpoint / every):
                pass
        else:
                check[checkpoint / every] = True
                print checkpoint


