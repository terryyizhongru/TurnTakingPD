import os
import sys

leftname = sys.argv[1]
filterfile = sys.argv[2]

for folder in ['BoundaryTone', 'EarlyLate', 'PictureNaming']:
    filename = leftname + folder + '.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    filterids = set()
    with open(filterfile, 'r') as f:
        for line in f:
            filterids.add(line.strip().split('.')[0])
    
    with open(filename.replace('.txt', '_filtered.txt'), 'w') as f:
        for line in lines:
            if line.split('\t')[0].split('.')[0] not in filterids:
                f.write(line)
                
                