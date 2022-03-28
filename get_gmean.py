#! /usr/bin/env python3

import sys
import statistics

res = {}
for line in open(sys.argv[1]):
    line = line.split(",")

    key = ",".join(line[:-1])
    val = float(line[-1])

    if key not in res:
        res[key] = []

    res[key].append(val)

for k in res:
    print("{:<48} -> {:<12}".format(k, statistics.geometric_mean(res[k])))
