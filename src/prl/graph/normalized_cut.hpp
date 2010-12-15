#ifndef PRL_GRAPH_NORMALIZED_CUT_HPP
#define PRL_GRAPH_NORMALIZED_CUT_HPP

#include <prl/global.hpp>
#include <prl/map.hpp>

#include <prl/macros_def.hpp>

namespace prl { 
  
  // using namespace boost::graph;

  /*
  template <typename K, typename G>
  map<typename G::vertex_descriptor, bool>
  normalized_cut(const G& g) {
    concept_assert((VertexAndEdgeListGraph<G>));
    typedef typename G::vertices_size_type size_type;  // need G::size_type
    // typedef typename G::index_type index_type
    typedef typename G::vertex_descriptor vertex_type;
    typedef typename G::edge_descriptor edge_type;
    // PRL_TYPEDEFS(G,(vertex_type)(index_type));
    
    // compute the adjacency matrix
    size_type n = num_vertices(M);
    typename K::symmetric_matrix w(zeros(n, n));
    typename property_map<G,vertex_index_t>::type index = get(vertex_index, g);
    foreach(edge_type e, edges(g)) {
      // w(e.source().index(), e.target().index()) = 1;
      // w(e.target().index(), e.source().index()) = 1;
      w(index[source(e,g)], index[target(e,g)]) = 1;
      w(index[target(e,g)], index[source(e,g)]) = 1;
    }

    // compute the relaxed solution
    typename K::vector relaxed = normalized_cut<K>(w);

    // compute the assignment
    std::size_t i = 0;
    map<vertex_type, bool> result;
    foreach(vertex_type v, vertices(g)) 
      result[v] = (relaxed[i++]>0);
    return result;
  }
  */

  //! \ingroup graph_algorithms
  template <typename LA>
  typename LA::vector_type
  normalized_cut(const typename LA::matrix_type& w) {
    concept_assert((LinearAlgebra<LA>));
    typename K::vector d = sum(w, 0);
    
    // compute the normalized Laplacian
    typename K::diagonal_matrix invd1_2 = 
      diag(1 / sqrt(d));
    typename K::symmetric_matrix normlap = 
      prod(invd1_2, typename K::matrix(prod(diag(d) - w, invd1_2)));

    // compute the top 2 eigenvectors
    typename K::matrix r;
    typename K::vector e;
    boost::tie(r,e) = eig(normlap);
    return prod(diag(sqrt(d)), column(r, 1));
  }

} // namespace prl

#include <prl/macros_undef.hpp>

#endif
