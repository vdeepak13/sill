#!/usr/bin/env python

import sys, os, time

if len(sys.argv) < 3:
    print "Usage: " + sys.argv[0] + " host-file niters" 
    exit(-1)

fvalue = open("values.txt", 'r')
fhosts = open(sys.argv[1], 'r')
niters = sys.argv[2]

values = fvalue.readlines()
hosts = fhosts.readlines()

for i in range(len(hosts)):
    value = values[i][:-1]
    host = hosts[i][:-1]
    # value host-name port host-file niters
    cmd = "../../debug/tests/distributed/test_pairwise_averaging " + value + " " + host + " " + sys.argv[2] + " " + niters + " > results/" + str(i) + ".txt 2> results/" + str(i) + ".log &"
    #print i, value, host
    time.sleep(0.1)
    print cmd
    os.system(cmd)



   




