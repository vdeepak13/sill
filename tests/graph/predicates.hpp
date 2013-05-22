#ifndef SILL_TEST_GRAPH_PREDICATES_HPP
#define SILL_TEST_GRAPH_PREDICATES_HPP

#include <fstream>
#include <cstdio>

#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>

#include <sill/macros_def.hpp>

//! Serializes and deserializes a value and verifies they are equal
template <typename T>
boost::test_tools::predicate_result
serialize_deserialize(const T& value) {
  char filename[L_tmpnam];
  tmpnam(filename);

  std::ofstream fout(filename, std::fstream::binary);
  sill::oarchive oa(fout);
  oa << value;
  fout.close();

  std::ifstream fin(filename, std::fstream::binary);
  sill::iarchive ia(fin);
  T value2;
  ia >> value2;
  fin.close();

  remove(filename);

  if(value != value2) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Serialization did not preserve the value [\n"
                     << value << "!="
                     << value2 << "]";
    return result;
  }
  return true;
}


template <typename V, typename Graph>
boost::test_tools::predicate_result
is_partial_vertex_order(const std::vector<V>& order_vec, const Graph& g) {
  if (order_vec.size() != g.num_vertices()) {
    boost::test_tools::predicate_result result(false);
    result.message() << "The partial order vector has invalid length: "
                     << order_vec.size() << "!=" << g.num_vertices();
    return result;
  }

  std::map<V, size_t> order;
  for (size_t i = 0; i < order_vec.size(); ++i) {
    if (!g.contains(order_vec[i])) {
      boost::test_tools::predicate_result result(false);
      result.message() << "Invalid vertex in the partial order: "
                       << order_vec[i];
      return result;
    }
    order[order_vec[i]] = i;
  }
  
  foreach(typename Graph::edge e, g.edges()) {
    if (!(order[e.source()] < order[e.target()])) {
      boost::test_tools::predicate_result result(false);
      result.message() << "The order does not respect the edge " << e;
      return result;
    }
  }

  return true;
}

#include <sill/macros_undef.hpp>

#endif
