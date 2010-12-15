#!/usr/bin/env python

import sys, os, time

# ./run_list_averaging.py localhosts10.txt 30 pushsum
# Expected result: 0.001270

if len(sys.argv) != 4:
    print "Usage: " + sys.argv[0] + " host-file niters algorithm" 
    exit(-1)

fvalue = open("values.txt", 'r')
fhosts = open(sys.argv[1], 'r')
niters = sys.argv[2]
algorithm = sys.argv[3]

values = fvalue.readlines()
hosts = fhosts.readlines()

for i in range(len(hosts)):
    value = values[i][:-1]
    host = hosts[i][:-1]
    # value host-name port host-file niters algorithm
    cmd = "../../debug/tests/distributed/test_list_averaging " + value + " " + host + " " + sys.argv[1] + " " + niters + " " + algorithm + " > results/" + str(i) + ".txt 2> results/" + str(i) + ".log &"
    #print i, value, host
    time.sleep(0.1)
    print cmd
    os.system(cmd)
