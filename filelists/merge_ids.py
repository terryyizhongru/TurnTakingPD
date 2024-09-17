import os
import sys

filename = sys.argv[1]
mergefile = sys.argv[2]

with open(filename, 'r') as f:
    lines = f.readlines()


        
mergeinfo = {}
for line in lines:
    key = line.split('\t')[0]
    if key in mergeinfo:
        print(key)
    mergeinfo[key] = line.split('\t')[1]

cnt = 0
with open(mergefile, 'r') as f:
    for line in f:
        key = line.split('\t')[0]
        if key in mergeinfo:
            cnt += 1
        mergeinfo[key] = line.split('\t')[1]



print(cnt)
with open(filename.replace('.txt', '_merged.txt'), 'w') as f:
    for k,v in mergeinfo.items():
        f.write(k + '\t' + v)
        
                
                