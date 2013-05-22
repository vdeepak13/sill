#!/usr/bin/python
import os
import string
import glob
import sys


if len(sys.argv) < 2:
  print "insufficient parameters. Need numparts!!"
  sys.exit()
#endif

# read the list of log files and get the network name, 
# number of processors, and number of partitions
dirlist = glob.glob(os.getcwd()+"/log_*")
dirname = os.getcwd()
networkname=dirname[(dirname.rfind("/")+1):len(dirname)]
numprocs=len(dirlist)
numparts=int(sys.argv[1])

# generate the "experiment prefix"
prefix=networkname+"\t"+str(numprocs)+"\t"+str(numparts)+"\t"

fout = open("cpuactivity.txt","w")
numsend=[0]*len(dirlist)
numrecv=[0]*len(dirlist)
runtime = 0;
for f in dirlist:
  fin = open(f,"r")
  tidx = 0
  procid = int(f[(f.rfind("_")+1):len(f)])
  started = False
  for line in fin:
    cmd = line[0:7]
    rawval = float(line[8:len(line)])
    if (cmd == "started"):
      fout.write(prefix+str(procid)+"\t"+str(rawval)+"\t"+"1\t1\t2\n")
      runtime = max(runtime, rawval);
    elif (cmd == "stopped"):
      fout.write(prefix+str(procid)+"\t"+str(rawval)+"\t"+"-1\t1\t2\n")   
      runtime = max(runtime, rawval)
    elif (cmd == "numsent"):
      numsend[procid]+=int(rawval)
    elif (cmd == "numrecv"):
      numrecv[procid]+=int(rawval)
    #end if
  #end for
#end for

fout = open("netactivity.txt","w")
for idx in range(len(numsend)):
  fout.write(prefix+str(idx)+"\t"+str(numsend[idx])+"\t"+str(numrecv[idx])+"\t1\t2\n")
#end for

# we need to extract the Energy and the loglikelihood from the runoutput.txt file
p=os.popen("grep Energy runoutput.txt")
energystring=p.read();
energystring=energystring.strip();
energy=float(energystring.rpartition(":")[2])

p=os.popen("grep Likelihood runoutput.txt")
likestring=p.read();
likestring=likestring.strip();
likelihood=float(likestring.rpartition(":")[2])

p=os.popen("grep Time runoutput.txt")
totalruntimestring=p.read();
totalruntimestring=totalruntimestring.strip();
totalruntime=float(totalruntimestring.rpartition(":")[2])

fout = open("runtime.txt","w")
fout.write(prefix+str(runtime)+"\t"+str(totalruntime)+"\t"+str(energy)+"\t"+str(likelihood)+"\t1\t2\n")
#end for
