#!/usr/bin/python

import re
import sys

if (len(sys.argv) != 3):
  print "Usage: acemarg2alchemy [src ace belief output] [dest alchemy output]"
  quit()
#endif

f = open(sys.argv[1], "r")
of = open(sys.argv[2], "w")
regex = re.compile("^.*__([0-9]*).*\[(.*)\]\s*$")
for line in f:
  m = regex.match(line)
  if (m):
    vals = m.group(2)
    # remove the commas
    vals = vals.replace(",","");
    of.write(m.group(1) + ", " + vals+"\n");
  #endif
#endfor
