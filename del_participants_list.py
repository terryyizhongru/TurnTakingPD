import sys
import os

del_list = '''1102
1114 
1115 
1120 
2132 
2220'''

del_list = [num.strip() for num in del_list.split("\n")]

del_dir = "/scratch/yzhong/Turntaking/"	

cnt = 0
# delete all wav filesin del_dir contains the number 
for num in del_list:
    for root, dirs, files in os.walk(del_dir):
        for file in files:
            if num in file:
                os.remove(os.path.join(root, file))
                print("Deleted: ", os.path.join(root, file))
                cnt += 1

print(cnt, "files deleted.")