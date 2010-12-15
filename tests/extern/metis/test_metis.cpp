#include <iostream>
#include <stdlib.h>


#include <metis/metis.hpp>

using namespace std;
using namespace metis;

int main(int argc, char** argv) {

  // number of vertices
  int nverts = 5;
  idxtype vweight[] = {1000, 2000, 3000, 4000, 5000}; 
  
  //int nedges = 12;

  idxtype xadj[] = {0,2,4,8,11,12};  

  idxtype adjncy[] = {2,3, 
                      2,3,
                      0,1,3,4,
                      0,1,2,
                      2};

  idxtype eweight[] = {1,1,
                       1,1,
                       1,1,1,1,
                       1,1,1,
                       1};
  

  /**
   * 0 No weights (vwgts and adjwgt are NULL) 
   * 1 Weights on the edges only (vwgts = NULL) 
   * 2 Weights on the vertices only (adjwgt = NULL) 
   * 3 Weights both on vertices and edges. 
   */
  int weightflag = 3;
  
  // 0 for C-style numbering starting at 0 (1 for fortran style)
  int numflag = 0;

  // the number of parts to cut into 
  int nparts = 2;
  
  // Options array (only care about first element if first element is zero
  int options[5]; options[0] = 0;

  // output argument number of edges cut
  int edgecut = 0;

  // output argument the array of assignments
  idxtype* part = new idxtype[nverts];


  // Cut the graph
  METIS_PartGraphKway(&(nverts), 
                      (idxtype*) xadj,
                      (idxtype*) adjncy,
                      (idxtype*) vweight,
                      (idxtype*) eweight,
                      &(weightflag),
                      &(numflag),
                      &(nparts),
                      options,
                      &(edgecut),
                      (idxtype*) part);

  for(int i = 0; i < nverts; ++i) {
    cout << part[i] << endl;
  }
  

  return EXIT_SUCCESS;
}
