#ifndef SILL_MODEL_FACTOR_GRAPH_PARTITIONING_HPP
#define SILL_MODEL_FACTOR_GRAPH_PARTITIONING_HPP

#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>

// PRL Includes
#include <sill/model/factor_graph_model.hpp>
#include <sill/model/mooij_kappen_derivatives.hpp>

#include <metis/metis.hpp>

// STL includes
#include <map>
#include <vector>
#include <set>
#include <queue>
#include <iostream>

// This include should always be last
#include <sill/macros_def.hpp>



namespace sill {

  template <typename F>
  class factor_graph_partition {
  public:
    enum algorithm {PMETIS, KMETIS, RANDOM, BFS};
    typedef factor_graph_model<F> factor_graph_type;
    typedef typename factor_graph_type::vertex_type vertex_type;
    typedef std::map<vertex_type, size_t> vertex2part_type;
    typedef std::set<vertex_type> vertex_set_type;
    typedef std::vector< vertex_set_type > part2vertex_type;
    typedef std::map<vertex_type, metis::idxtype> vertex_weight_map;

  private:
    size_t nparts_;
    algorithm alg_;
    bool weighted_;
    vertex2part_type vertex2part_;
    part2vertex_type part2vertex_;

  public:

    /**
     * Construct a factor graph partitioning with nparts using the
     * algorithm passed in as a string "pmetis", "kmetis", "random"
     * "bfs", and an optional pointer to a vertex weight map.
     */
    factor_graph_partition(const factor_graph_type& fg,
                           size_t nparts,
                           std::string alg_str,
                           bool weighted,
                           const vertex_weight_map* vertex_weights = NULL) {
      algorithm alg = string2alg(alg_str);
      partition(fg, nparts, alg, weighted, vertex_weights);
    } // end of constructor


    /**
     * Construct a factor graph partitioning with nparts using the
     * algorithm specified and an optional pointer to a vertex weight
     * map.
     */
    factor_graph_partition(const factor_graph_type& fg,
                           size_t nparts,
                           algorithm alg,
                           bool weighted,
                           const vertex_weight_map* vertex_weights = NULL) {
      partition(fg, nparts, alg, weighted, vertex_weights);
    } // end of constructor


    /**
     * cut the actual factor graph
     */
    void partition(const factor_graph_type& fg,
                   size_t nparts,
                   algorithm alg,
                   bool weighted,
                   const vertex_weight_map* vertex_weights = NULL) {
      nparts_ = nparts;
      alg_ = alg;
      weighted_ = weighted;

      vertex2part_.clear();
      part2vertex_.clear();
      part2vertex_.resize(nparts_);


      if(nparts_ == 1) { // partition into one partition (i.e. don't
                         // partition)
        foreach(vertex_type vert, fg.vertices()){
          vertex2part_[vert] = 0;
          part2vertex_[0].insert(vert);
        }
      } else {      
        // do partitioning
        switch(alg_) {
        case PMETIS:
          std::cout << "Running pmetis cut" << std::endl;
          metis_cut(fg, vertex_weights);
          break;
        case KMETIS:
          std::cout << "Running kmetis cut" << std::endl;
          metis_cut(fg, vertex_weights);
          break;
        case RANDOM:
          std::cout << "Running random cut" << std::endl;
          random_cut(fg);
          break;
        case BFS:
          std::cout << "Running breadth first search (bfs) cut" << std::endl;
          bfs_cut(fg);
          break;
        default:
          std::cout << "Unavailable Cut!" << std::endl;
          assert(false);
        }
      }
      std::cout << "Cutting into " << nparts << " partitions" << std::endl; 

      // Verify the partition
//       foreach(vertex_set_type& vset, part2vertex_) {
//         assert(!vset.empty());
//       }

    } // end of partition

    /**
     * convert an algorithm string into an algorithm identifier
     */
    algorithm string2alg(std::string alg_str) const {
      typedef std::map<std::string, algorithm> map_type;
      map_type algmap;
      algmap["pmetis"] = PMETIS;
      algmap["kmetis"] = KMETIS;
      algmap["random"] = RANDOM;
      algmap["bfs"] = BFS;
      typedef typename map_type::const_iterator iterator;
      iterator iter = algmap.find(alg_str);
      if(iter != algmap.end()) {
        return iter->second;
      } else {
        // Default to kmetis
        return KMETIS;
      }
    }


    /**
     * Get the number of parts
     */
    size_t number_of_parts() const {
      return nparts_;
    }


    /**
     * Get the part for a particular vertex
     */
    size_t vertex2part(const vertex_type& v) const {
      typedef typename vertex2part_type::const_iterator iterator;
      iterator iter = vertex2part_.find(v);
      assert(iter != vertex2part_.end());
      return iter->second;
    }

    /**
     * Get all the vertices in a particular part
     */
    const vertex_set_type& part2vertices(size_t part) const {
      assert(part < part2vertex_.size());
      return part2vertex_[part];
    }

    /**
     * Print the partitioning to outputstream
     */
    void print(std::ostream& out) const {
      typedef std::pair<vertex_type, size_t> pair;
      foreach(pair p, vertex2part_) {
        out << p.second << std::endl;
      }
    } // end of print partitioning


    /**
     * Compute the balance score
     */
    double balance_score(const factor_graph_type& fg,
                         bool weighted) const {
      double maxscore = 0.0;
      foreach(const vertex_set_type& vset, part2vertex_) {
        double score = 0.0;
        if(weighted) {
          foreach(const vertex_type& v, vset)
            score += fg.work_per_update(v);
        } else {
          score = vset.size();
        }
        maxscore = std::max(maxscore, score);
      }
      return maxscore;
    }

    /**
     * Print the weighted balance
     */
    void print_balance(const factor_graph_type& fg,
                       bool weighted,
                       std::ostream& out) const {
      foreach(const vertex_set_type& vset, part2vertex_) {
        double score = 0.0;
        if(weighted) {
          foreach(const vertex_type& v, vset)
            score += fg.work_per_update(v);
        } else {
          score = vset.size();
        }
        out << score << ", ";
      }
      out << std::endl;
    } // end of print balance



    /**
     * Compute the total communcation score
     */
    double total_comm_score(const factor_graph_type& fg) const {
      double score = 0.0;
      foreach(vertex_type u, fg.vertices()) {
        size_t upart = vertex2part_[u];
        foreach(vertex_type v, fg.neighbors(u)){
          if(upart != vertex2part_[v]) score += 1.0;
        }
      }
      return score;
    }


    /**
     * Compute the max communication score
     */
    double max_comm_score(const factor_graph_type& fg) const {
      typedef typename vertex2part_type::const_iterator iterator;
      double maxscore = 0.0;
      size_t upart = 0;
      foreach(const vertex_set_type& vset, part2vertex_) {
        double score = 0.0;
        foreach(vertex_type u, vset) {
          foreach(vertex_type v, fg.neighbors(u)){
           iterator iter = vertex2part_.find(v);
           assert(iter != vertex2part_.end());
           if(iter->second != upart) score += 1.0;
          }
        }
        maxscore = std::max(maxscore, score);
        upart++;
      }
      return maxscore;
    } // end of max_comm_score

    /**
     * Print the communication score for each block
     */
    void print_comm_score(const factor_graph_type& fg,
                          std::ostream& out) const {
      typedef typename vertex2part_type::const_iterator iterator;
      size_t upart = 0;
      foreach(const vertex_set_type& vset, part2vertex_) {
        double score = 0.0;
        foreach(vertex_type u, vset) {
          foreach(vertex_type v, fg.neighbors(u)){
            iterator iter = vertex2part_.find(v);
            assert(iter != vertex2part_.end());
            if(iter->second != upart) score += 1.0;
          }
        }
        out << score << ", ";
        upart++;
      }
      out << std::endl;
    } // end of max_comm_score


  private:
    
    // do the actual metis cut
    void metis_cut(const factor_graph_type& fg,
                   const vertex_weight_map* vertex_weights) {
      // Determine parameters needed to construct the partitioning
      int numverts = static_cast<int>(fg.num_vertices());
      assert(numverts > 0);
      // Compute the number of edges 
      int numedges = 0;
      foreach(vertex_type v, fg.vertices()) {
        numedges += fg.num_neighbors(v);
      }

      // allocate metis data structures
      metis::idxtype* vweight = new metis::idxtype[numverts];
      assert(vweight != NULL);    
      metis::idxtype* xadj = new metis::idxtype[numverts + 1];
      assert(xadj != NULL);
      metis::idxtype* adjacency = new metis::idxtype[numedges];
      assert(adjacency != NULL);
      metis::idxtype* eweight = NULL;
      //       if(weighted_) {
      //         eweight = new idxtype[numedges];
      //         assert(eweigth != NULL);
      //       }
      metis::idxtype* res = new metis::idxtype[numverts];   
      assert(res != NULL);

      // Pass through vertices filling in the metis data structures
      size_t offset = 0;
      foreach(vertex_type u, fg.vertices()) {
        // Get the vertex id
        size_t u_id = fg.vertex2id(u);
        assert(u_id < static_cast<size_t>(numverts));
        // Update vertex weight
        if(weighted_) {
          if(vertex_weights != NULL) {
            typedef typename vertex_weight_map::const_iterator iterator;
            iterator iter = vertex_weights->find(u);
            assert(iter != vertex_weights->end());
            vweight[u_id] = iter->second;
          }else {
            vweight[u_id] = fg.work_per_update(u);
          }
        } else { 
          vweight[u_id] = 1;
        }
        // Update the offset
        xadj[u_id] = offset;
        // Fill the the adjacency data
        foreach(size_t vid, fg.neighbor_ids(u)) {
          adjacency[offset] = vid;
          assert(adjacency[offset] >= 0);
          //           if(weighted_) eweight[offset] = 1;
          // Move to the next offset
          offset++;
          assert(offset >= 0);
        }
      }

      // Set the last entry in xadj to the end of the adjacency array
      xadj[numverts] = offset;
      
      // Set additional metis flags
      /**
       * 0 No weights (vwgts and adjwgt are NULL) 
       * 1 Weights on the edges only (vwgts = NULL) 
       * 2 Weights on the vertices only (adjwgt = NULL) 
       * 3 Weights both on vertices and edges. 
       */
      int weightflag = 2;
      // 0 for C-style numbering starting at 0 (1 for fortran style)
      int numflag = 0;
      // the number of parts to cut into
      int nparts = nparts_;     
      // Options array (only care about first element if first element
      // is zero
      int options[5]; options[0] = 0;
      // output argument number of edges cut
      int edgecut = 0;
      if(alg_ == KMETIS) {
        // Call kmetis
        metis::METIS_PartGraphKway(&(numverts), 
                            xadj,
                            adjacency,
                            vweight,
                            eweight,
                            &(weightflag),
                            &(numflag),
                            &(nparts),
                            options,
                            &(edgecut),
                            res);
      } else {
        // Call pmetis
        metis::METIS_PartGraphRecursive(&(numverts), 
                                 xadj,
                                 adjacency,
                                 vweight,
                                 eweight,
                                 &(weightflag),
                                 &(numflag),
                                 &(nparts),
                                 options,
                                 &(edgecut),
                                 res);
      } // end of if
      // destroy all unecessary data structures except res
      if(xadj != NULL) delete [] xadj;
      if(adjacency != NULL) delete [] adjacency;
      if(vweight != NULL) delete [] vweight;
      if(eweight != NULL) delete [] eweight;
      // process the final results
      for(int vid = 0; vid < numverts; ++vid) {
        vertex_type v = fg.id2vertex(vid);
        vertex2part_[v] = res[vid];
        part2vertex_[res[vid]].insert(v);
      }    
      // Delete the result array
      if(res != NULL) delete [] res;
    } // end of cut_metis


    
    /**
     * Very simple random cut.  Equal number of vertices in each set. 
     */
    void random_cut(const factor_graph_type& fg) {
      std::vector<vertex_type> vertices(fg.num_vertices());
      size_t i = 0;
      foreach(const vertex_type& v, fg.vertices()) vertices[i++] = v;
      // Shufffle the vertices
      std::random_shuffle(vertices.begin(), vertices.end());
      // assign the vertices to blocks
      size_t partid = 0;
      foreach(vertex_type v, vertices) {
        vertex2part_[v] = partid;
        part2vertex_[partid].insert(v);
        if(++partid >= nparts_) partid = 0;
      }
    } // end of random cut


    /**
     * Do a simple bfs cut of the graph
     */
    void bfs_cut(const factor_graph_type& fg) {
      // create a list of all unassigned variables
      std::set<vertex_type> unassigned;
      // initialize the unassigned vertices
      foreach(const vertex_type &v, fg.vertices()) {
        unassigned.insert(v);
      }
      // Compute the partition size
      size_t maxpartsize = (unassigned.size() / nparts_) + 1;
      size_t partid = 0;
      while(!unassigned.empty()) {  
        std::list<vertex_type> queue;    // Breadth first queue 
        std::set<vertex_type>  visited;  // Set of visited vertices
        // While the task is still too small and their remains
        // unassigned vertices
        while(part2vertex_[partid].size() < maxpartsize 
              && !unassigned.empty()) {
          if(queue.empty()) { 
            queue.push_front(*unassigned.begin());
            visited.insert(*unassigned.begin());
          }
          assert(!queue.empty());
          // Pop the first element off the queue 
          vertex_type v = queue.front(); queue.pop_front();
          assert(partid < nparts_);
          // Add the element to the task
          part2vertex_[partid].insert(v);
          vertex2part_[v] = partid;
          // Remove the vertex from the set of unassigned vertices
          unassigned.erase(v); 
          // Add all its unassigned and unvisited neighbors to the queue
          foreach(vertex_type u, fg.neighbors(v)) {
            if(unassigned.find(u) != unassigned.end() &&
               visited.find(u) == visited.end()) {
              queue.push_back(u);
              visited.insert(u);
            }
          } // end of add neighbors for loop
        } // End of block build foor loop
        // move to the next part
        partid++;
      }// end of outer while loop
    } // end of bfs cut
  }; // End of factor graph partition


                       




//   template<typename F>
//   void SliceGraphRandom(
//     factor_graph_model<F> &model,     // graph
//     int numowners,                    // number of partitions
//     std::vector<std::set<typename factor_graph_model<F>::vertex_type> > &owner2vertex, // output assignments
//     std::map<typename factor_graph_model<F>::vertex_type, uint32_t> &vertex2owner) {  // output assignments
//       // create a list of all unassigned variables
//       typedef typename factor_graph_model<F>::vertex_type vertex_type;
//       std::set<vertex_type> unassigned;
//       std::vector<vertex_type> unassignedv;

//       foreach(const vertex_type &v, model.vertices()) {
//         unassignedv.push_back(v);
//       }
//       random_shuffle(unassignedv.begin(), unassignedv.end());
//       foreach(const vertex_type &v, unassignedv) {
//         unassigned.insert(v);
//       }
     
//       int maxblocksize = unassigned.size() / numowners + 1;
//       while(!unassigned.empty()) {  
//         std::list<vertex_type> queue;    // Breadth first queue 
//         std::set<vertex_type>  visited;  // Set of visited vertices
//         std::set<vertex_type> verticesinthisblock;
//         // While the task is still too small and their remains unassigned
//         // vertices
//         int curblocksize = 0;
//         while(curblocksize < maxblocksize && !unassigned.empty()) {
//           if(queue.empty()) { 
//             queue.push_front(*unassigned.begin());
//             visited.insert(*unassigned.begin());
//           }
//           assert(!queue.empty());
//           // Pop the first element off the queue 
//           vertex_type v = queue.front(); queue.pop_front();
//           // Add the element to the task
//           verticesinthisblock.insert(v);
//           curblocksize++;
          
//           // Remove the vertex from the set of unassigned vertices
//           unassigned.erase(v); 
//           // Add all its unassigned and unvisited neighbors to the queue
//           foreach(vertex_type u, model.neighbors(v)) {
//             if(unassigned.find(u) != unassigned.end() &&
//               visited.find(u) == visited.end()) {
//               queue.push_back(u);
//               visited.insert(u);
//             }
//           } // end of add neighbors for loop
//           queue.clear();
//         } // End of block build foor loop
//         owner2vertex.push_back(verticesinthisblock);
//         int ownerid = owner2vertex.size() - 1;
//         assert(ownerid < numowners);
//         foreach(const vertex_type &v, verticesinthisblock) {
//           vertex2owner[v] = ownerid;
//         }
//       }
//     } 
  
  
//   // mostly copied from task_manager in old paraml code
//   /** cuts up the factor graph using repeated BFS's and fills in
//       the structures vertex2owner, block2owner_ and total_block_count_
//   */
//   template <typename F>
//   void SliceGraphBFS(
//     factor_graph_model<F> &model,     // graph
//     int numowners,                    // number of partitions
//     std::vector<std::set<typename factor_graph_model<F>::vertex_type> > &owner2vertex, // output assignments
//     std::map<typename factor_graph_model<F>::vertex_type, uint32_t> &vertex2owner) {  // output assignments
    
//     typedef typename factor_graph_model<F>::vertex_type vertex_type;
//     // create a list of all unassigned variables
//     std::set<vertex_type> unassigned;
    
//     foreach(const vertex_type &v, model.vertices()) {
//       unassigned.insert(v);
//     }
    
//     int maxblocksize = unassigned.size() / numowners + 1;
//     while(!unassigned.empty()) {  
//       std::list<vertex_type> queue;    // Breadth first queue 
//       std::set<vertex_type>  visited;  // Set of visited vertices
//       std::set<vertex_type> verticesinthisblock;
//       // While the task is still too small and their remains unassigned
//       // vertices
//       int curblocksize = 0;
//       while(curblocksize < maxblocksize && !unassigned.empty()) {
//       if(queue.empty()) { 
//           queue.push_front(*unassigned.begin());
//           visited.insert(*unassigned.begin());
//         }
//         assert(!queue.empty());
//         // Pop the first element off the queue 
//         vertex_type v = queue.front(); queue.pop_front();
//         // Add the element to the task
//         verticesinthisblock.insert(v);
//         curblocksize++;
        
//         // Remove the vertex from the set of unassigned vertices
//         unassigned.erase(v); 
//         // Add all its unassigned and unvisited neighbors to the queue
//         foreach(vertex_type u, model.neighbors(v)) {
//           if(unassigned.find(u) != unassigned.end() &&
//             visited.find(u) == visited.end()) {
//             queue.push_back(u);
//             visited.insert(u);
//           }
//         } // end of add neighbors for loop
//       } // End of block build foor loop
//       owner2vertex.push_back(verticesinthisblock);
//       int ownerid = owner2vertex.size() - 1;
//       assert(ownerid < numowners);
//       foreach(const vertex_type &v, verticesinthisblock) {
//         vertex2owner[v] = ownerid;
//       }
//     }
//   }
  
  
//   template <typename F>
//   bool SliceGraph(partition_algorithm partalgo, 
//         factor_graph_model<F> &model,     // graph
//         int numowners,                    // number of partitions
//         std::vector<std::set<typename factor_graph_model<F>::vertex_type> > &owner2vertex, // output assignments
//         std::map<typename factor_graph_model<F>::vertex_type, uint32_t> &vertex2owner) { // output assignments
//     switch(partalgo) {
//       case PARTITION_METIS:
//         return SliceGraphMetis<F>(model,numowners,owner2vertex,vertex2owner);
//       case PARTITION_RANDOM:
//         SliceGraphRandom<F>(model,numowners,owner2vertex,vertex2owner);
//         return true;
//       case PARTITION_BFS:
//         SliceGraphBFS<F>(model,numowners,owner2vertex,vertex2owner);
//         return true;
//       default:
//         return false;
//     }
//   }


}
#endif
