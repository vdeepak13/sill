// This file has not been tested yet

#ifndef PRL_MARKOV_NETWORK_XML_HPP
#define PRL_MARKOV_NETWORK_XML_HPP

#include <prl/model/decomposable.hpp>
#include <prl/archive/xml_iarchive.hpp>
#include <prl/archive/xml_oarchive.hpp>
#include <prl/archive/xml_tag.hpp>

#include <prl/macros_def.hpp>

namespace prl {
  
  template <typename NodeF, typename EdgeF>
  const char* xml_tag(pairwise_markov_network<NodeF, EdgeF>*) {
    return "markov_network";
  }

  template <typename NodeF, typename EdgeF>
  xml_ostream& operator<<(xml_ostream& out,
                          const pairwise_markov_network<NodeF, EdgeF>& mn) {
    out.register_variables(mn.nodes());
    out.save_begin("markov_network");
    out.add_attribute("factor_type", xml_tag((NodeF*)NULL));
    
    // Save the factors
    foreach(const NodeF& factor, mn.node_factors())
      out << factor;
    foreach(const EdgeF& factor, mn.edge_factors())
      out << factor;

    out.save_end();
    return out;
  }

  // for now, we only support reading into a markov network with same
  // node and edge factor types
  template <typename F>
  xml_istream& operator>>(xml_istream& in, pairwise_markov_network<F>& mn) {
    in.load_begin("markov_network");
    mn.clear();
    
    while(in.has_next()) {
      F f;
      in >> f;
      mn.add_factor(f);
    }
    
    in.load_end();
    return in;
  }

} // namespace prl

#include <prl/macros_undef.hpp>

#endif
