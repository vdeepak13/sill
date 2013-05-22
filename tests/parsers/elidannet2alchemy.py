#!/usr/bin/python

import re
import sys
import math

class Clique:
  name = ""
  numvars = 0
  v = []
  meas = 0
  

if (len(sys.argv) != 3):
  print "Usage: acemarg2alchemy [src elidan net file] [dest alchemy file]"
  quit()
#endif

PARSEVARIABLE=0
PARSECLIQUE=1
PARSEMEASURE=2
PARSECLIQUE2MEASURE=3
parsestate = 0

f = open(sys.argv[1], "r")
of = open(sys.argv[2], "w")
varname = []
cliques = []
measures = []
of.write("variables:\n");

for line in f:
  line = line.strip()
  if (line == "@Variables"):
    parsestate=PARSEVARIABLE
    continue
  elif (line == "@Cliques"):
    parsestate=PARSECLIQUE
    continue
  elif (line == "@Measures"):
    parsestate=PARSEMEASURE
    continue
  elif (line == "@CliqueToMeasure"):
    parsestate=PARSECLIQUE2MEASURE
    of.write("factors:\n");
    continue
  elif (line == "@DirectedMeasures"):
    break;
  elif (line == "@End" or line == ""):
    continue
  #endif
  
  if (parsestate == PARSEVARIABLE):
    line=line.strip()
    of.write(line + '\n')
    varname.append(line.split('\t')[0])
  elif (parsestate == PARSECLIQUE):
    curclique = Clique();
    linesplit=line.split('\t')
    curclique.name=linesplit[0].strip()
    curclique.numvars=int(linesplit[1])
    linesplit[2] = linesplit[2].strip()
    curclique.v=map(lambda x: int(x), linesplit[2].split(' '))
    if (len(curclique.v) != curclique.numvars):
      print curclique.v
      print curclique.numvars
    #endif
    cliques.append(curclique)
  elif (parsestate == PARSEMEASURE):
    linesplit=line.strip().split('\t')
    linesplit[3] = linesplit[3].strip()
    tologform = ' '.join(map(lambda x:str(math.log(float(x))),linesplit[3].split(' ')))
    measures.append(tologform)
  #endif
  elif (parsestate == PARSECLIQUE2MEASURE):
    linesplit=line.strip().split('\t')
    factorid=int(linesplit[0])
    measid = int(linesplit[1])
    varnamelist=map(lambda x: varname[x], cliques[factorid].v)
    of.write(" / ".join(varnamelist) + " // " + measures[measid] + "\n");
#endfor
