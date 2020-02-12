#!/usr/bin/env python3

import os
import sys

n = 512
if len(sys.argv) > 1:
    n = sys.argv[1]

# Remove bbc.txt if exists
if os.path.isfile("bbc.txt"):
    os.remove("bbc.txt")

# execute branch_divergence binary
res = os.popen("./branchDivergence " + str(n)).read()
print(res)
cols = res.split('\n')[-2].split(',')
n = int(cols[0])
rand_sum = int(cols[1])

# Read Basic Block Frequencies
bbc = open("bbc.txt",'r')
for line in bbc:
    pass
lastline = line
cols = lastline.split(',')

active_thread_count = int(cols[12])
loop_header_count = int(cols[13])
loop_body_count = int(cols[14])
loop_body_if = int(cols[15])
loop_body_else = int(cols[16])

# Check result
error = False
if (n != active_thread_count):
    print("n not matching with entry count of kernel")
    error = True
if (loop_header_count != rand_sum + n):
    print("loop header count not matching sum + n")
    error = True
if (loop_body_count != rand_sum):
    print("loop body count not matching sum")
    error = True
if (loop_body_if + loop_body_else != loop_body_count):
    print("if and else branch not matching loop body count")
    error = True

if error:
    print("n: " + str(n))
    print("sum: " + str(rand_sum))
    print("active_thread_count: " + str(active_thread_count))
    print("loop_header_count: " + str(loop_header_count))
    print("loop_body_count: " + str(loop_body_count))
    print("loop_body_if: " + str(loop_body_if))
    print("loop_body_else: " + str(loop_body_else))
    sys.exit(1)
else:
    print("Test passed!")
