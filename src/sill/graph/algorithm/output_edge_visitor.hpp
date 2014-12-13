#ifndef SILL_OUTPUT_EDGE_VISITOR
#define SILL_OUTPUT_EDGE_VISITOR

#include <sill/global.hpp>

namespace sill {

  //! A simple edge visitor that writes edges to an output iterator.
  //! \ingroup graph_types
  template <typename OutIt>
  class output_edge_visitor {
    OutIt out;
  public:
    output_edge_visitor(OutIt out) : out(out) { }
    template<typename Graph>
    void operator()(typename Graph::edge e, const Graph&) {
      out = e; ++out;
    }
  };

  //! A convenience function for constructing output_edge_visitor objects.
  //! \relates output_edge_visitor
  template <typename OutIt>
  output_edge_visitor<OutIt> make_output_edge_visitor(OutIt out) {
    return output_edge_visitor<OutIt>(out);
  }

}

#endif
