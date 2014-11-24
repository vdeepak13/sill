#ifndef SILL_KIKUCHI_REGION_GRAPH_HPP
#define SILL_KIKUCHI_REGION_GRAPH_HPP

#include <vector>
#include <iterator> // for back_inserter

#include <set>
#include <map>
#include <sill/model/region_graph.hpp>
#include <sill/range/algorithm.hpp>
#include <sill/functional/size_greater.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Constructs a valid region graph using Kikuchi method.
   * This method computes regions closed under the intersection.
   * Note that vertex and edge properties are default-initialized
   * (if these are factors, the user may need to change them to 1).
   *
   * \ingroup inference
   */
  template <typename Node, typename VP, typename EP>
  void kikuchi(const std::vector< std::set<Node> >& root_clusters,
               region_graph<Node, VP, EP>& rg) {
    std::set< std::set<Node> > clusters;
    set_index<std::set<Node>, size_t> index;
    // index: queue of sets for the purpose of computing the intersections
    rg.clear();

    // add the root clusters
    foreach(const std::set<Node>& cluster, root_clusters) {
      assert(!cluster.empty());
      if (!clusters.count(cluster)) {
        size_t r = rg.add_region(cluster);
        index.insert(cluster, r);
        clusters.insert(cluster);
      }
    }

    // compute closure under intersections
    while(!index.empty()) {
      size_t r = index.front();
      std::set<Node> cluster = rg.cluster(r);
      std::vector<size_t> regions;
      index.find_intersecting_sets(cluster, std::back_inserter(regions));
      foreach(size_t r2, regions) {
        std::set<Node> new_cluster = set_intersect(cluster,rg.cluster(r2));
        if (!clusters.count(new_cluster)) {
          size_t new_region = rg.add_region(new_cluster);
          clusters.insert(new_cluster);
          index.insert(new_cluster, new_region);
        }
      }
      index.remove(cluster, r);
    }

    // add the edges
    std::vector<size_t> regions(rg.num_vertices());
    sill::copy(rg.vertices(), regions.begin());
    sill::sort(regions,
              typename region_graph<Node,VP,EP>::cluster_size_less(&rg));

    for(int i = regions.size() - 1; i >= 0; i--) {
      size_t r = regions[i];
      std::set<size_t> ancestors; // all ancestors of the region r
      std::set<Node> cluster = rg.cluster(r);
      for(size_t j = i+1; j < regions.size(); j++) {
        size_t s = regions[j];
        if(!ancestors.count(s) && includes(rg.cluster(s), cluster)) {
          rg.add_edge(s, r);
          ancestors.insert(s);
          ancestors = set_union(ancestors, rg.ancestors(s));
        }
      }
    }

//     using namespace std;
//     cout << rg << endl;
//     cout << regions << endl;

    rg.update_counting_numbers();
  }

}

#include <sill/macros_undef.hpp>

#endif
