#!/usr/bin/env python

import sys, os, time

if len(sys.argv) < 5:
    print "Usage: " + sys.argv[0] + " host-file port chord-port niters" 
    exit(-1)

fvalue = open("values.txt", 'r')
fhosts = open(sys.argv[1], 'r')
port   = sys.argv[2]
cport  = sys.argv[3]
niters = sys.argv[4]

values = fvalue.readlines()
hosts = fhosts.readlines()

for i in range(len(hosts)):
    value = values[i][:-1]
    host = hosts[i][:-1]
    # value host-name port chord-port niters
    cmd = "nohup ssh " + host + " ~/prl/trunk/debug/tests/distributed/test_chord_pairwise_averaging " + value + " " + host + " " + port + " " + cport +" " + niters + " > results/" + str(i) + ".txt 2> results/" + str(i) + ".log &"
    #print i, value, host
    time.sleep(0.1)
    print cmd
    os.system(cmd)

# ./run_chord_pairwise_averaging.py values16.txt hosts16.txt 20000 10000 10
# should converge to 0.1328
