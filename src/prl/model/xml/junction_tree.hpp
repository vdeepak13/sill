#ifndef PRL_JUNCTION_TREE_XML_HPP
#define PRL_JUNCTION_TREE_XML_HPP

#include <prl/model/junction_tree.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  template <typename Node, typename VP, typename EP>
  void referenced_variables(const junction_tree<Node, VP, EP>& jt, domain& vars)
  {
    foreach(const domain& clique, jt.cliques())
      vars.insert(clique);
  }

  template <typename Node, typename VP, typename EP>
  const char* xml_tag(junction_tree<Node, VP, EP>*) {
    return "junction_tree";
  }

  template <typename Node, typename VP, typename EP>
  void write_xml(std::ostream& out, const junction_tree<Node, VP, EP>& jt, 
                 indent ind) {
    using std::endl;
    out << ind << "<junction_tree "
        << "node=\"" << xml_tag((Node*)NULL) << "\" "
        << "vertex_property=\"" << xml_tag((VP*)NULL) << "\" "
        << "edge_property=\"" << xml_tag((EP*)NULL) << "\""
        << ">" << endl;
    
    // todo: finish up

    out << ind << "</junction_tree>" << endl;
  }

  template <typename Node, typename VP, typename EP>
  void read_xml(xmlpp::TextReader& reader, junction_tree<Node, EP, VP>& f,
                const map<size_t, variable*>& var_map) {
    assert(false); // not implemented yet
  }

}

#include <prl/macros_undef.hpp>

#endif
