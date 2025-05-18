import os
import json
from pathlib import Path

known_interactions = open("graphs_subset2\graph_interactions.txt" , "r" , encoding="utf-8")
expected = open("output_compare_high.txt" , "r" , encoding="utf-8")
unexpected = open("output_compare_low.txt" , "r" , encoding="utf-8")

drugs = []
known = []
correct = []
false = []
directory = "drugs_split"

i = 0
for filename in os.listdir(directory):
    i = i + 1
    str = filename.removesuffix(".json")
    drugs.append(str)

rows, cols = (i, i)
known = [[0]*cols]*rows
high = [[0]*cols]*rows
low  = [[0]*cols]*rows

# for j in range(i):
#     for k in range(i):
#         known[j][k] = ""
correct_count = 0
false_count = 0

for line in known_interactions:
    res = line.split("|||")
    #print(res[2])
    known[drugs.index(res[0])][drugs.index(res[1])] = res[2].strip("\n")

for line in expected:
    res = line.split("|||")
    res[0] = res[0].removesuffix(",")
    (res[0])
    try:
        high[drugs.index(res[0])][drugs.index(res[1])] = res[2].strip("\n")
        if high[drugs.index(res[0])][drugs.index(res[1])] == known[drugs.index(res[0])][drugs.index(res[1])]:
            correct_count = correct_count + 1
    except:
        pass


for line in unexpected:
    res = line.split("|||")
    try:
        low[drugs.index(res[0])][drugs.index(res[1])] = res[2].strip("\n")
        if low[drugs.index(res[0])][drugs.index(res[1])] == known[drugs.index(res[0])][drugs.index(res[1])]:
            false_count = false_count + 1
    except:
        pass

print(correct_count)
print(false_count)