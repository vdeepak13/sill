#ifndef PRL_BETHE_REGION_GRAPH_HPP
#define PRL_BETHE_REGION_GRAPH_HPP

#include <vector>

#include <set>
#include <map>
#include <prl/model/region_graph.hpp>

#include <prl/macros_def.hpp>

namespace prl {
  
  /**
   * Constructs a valid region graph that corresponds to Bethe free energey
   * approximation.
   * Note that vertex and edge properties are default-initialized
   * (if these are factors, the user may need to change them to 1).
   *
   * \ingroup inference
   */
  template <typename Node, typename VP, typename EP>
  void bethe(const std::vector< std::set<Node> >& clusters,
             region_graph<Node,VP,EP>& rg) {
    std::map<Node, size_t> node_region; // single-node clusters
    rg.clear();

    // add all the singletons
    foreach(const std::set<Node>& cluster, clusters) {
      foreach(Node node, cluster)
        if (!node_region.count(node)) {
          size_t r = rg.add_region(make_domain(node));
          node_region[node] = r;
        }
    }
    
    // add the root clusters
    foreach(const std::set<Node>& cluster, clusters) {
      size_t r = rg.add_region(cluster);
      foreach(Node node, cluster)
        rg.add_edge(r, node_region[node]);
    }

    rg.update_counting_numbers();
  }

  /**
   * Constructs a valid region graph that corresponds to Bethe free energy
   * approximation. The singleton clusters are grouped, as specified.
   * Note that vertex and edge properties are default-initialized
   * (if these are factors, the user may need to change them to 1).
   * 
   * @param singletons
   * Specifies the partitioning of nodes in the child regions. 
   * The union of the singletons must be equal to the union of the clusters.
   *      
   * \ingroup inference
   */
  template <typename Node, typename VP, typename EP>
  void bethe(const std::vector< std::set<Node> >& clusters,
             const std::vector< std::set<Node> >& singletons,
             region_graph<Node,VP,EP>& rg) {
    // the singleton region associated with an individual node
    std::map<Node, size_t> node_region; 
    rg.clear();
    
    // add all the singletons
    foreach(const std::set<Node>& singleton, singletons) {
      size_t r = rg.add_region(singleton);
      foreach(Node node, singleton) {
        assert(!node_region.count(node)); // proper partition?
        node_region[node] = r;
      }
    }

    // add the root clusters
    foreach(const std::set<Node>& cluster, clusters) {
      size_t r = rg.add_region(cluster);
      foreach(Node node, cluster) {
        assert(node_region.count(node));
        rg.add_edge(r, node_region[node]);
      }
    }

    // check that all singletons intersect some root cluster
    typedef std::pair<Node, size_t> node_vertex_pair;
    foreach(node_vertex_pair& p, node_region)
      assert(rg.in_degree(p.second) > 0);

    rg.update_counting_numbers();
  }

}


#include <prl/macros_undef.hpp>

#endif
