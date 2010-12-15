#ifndef PRL_TABLE_FACTOR_XML_HPP
#define PRL_TABLE_FACTOR_XML_HPP

#include <iosfwd>

#include <prl/factor/table_factor.hpp>
#include <prl/archive/xml_iarchive.hpp>
#include <prl/archive/xml_oarchive.hpp>
#include <prl/archive/xml_tag.hpp>

namespace prl {

  template <typename Table>
  const char* xml_tag(table_factor<Table>*) { 
    return "table_factor"; 
  }

  template <typename Table>
  xml_oarchive& operator<<(xml_oarchive& out, const table_factor<Table>& f) {
    out.register_variables(f.arguments());
    out.save_begin("table_factor");
    out.add_attribute("storage", xml_tag((typename Table::value_type*)NULL));
    // I would expect "sparse" or "dense" here
    out << make_nvp("arg_list", f.arg_list());
    out.write_range("values", f.values());
    out.save_end();
    return out;
  }
  
  template <typename Table>
  xml_iarchive& operator>>(xml_iarchive& in, table_factor<Table>& f) {
    typedef typename Table::value_type value_type;
    in.load_begin("table_factor");
    finite_var_vector vars;
    in >> make_nvp("arg_list", vars);
    f = table_factor<Table>(vars, 1);
    in.read_range("values", f.values());
    in.load_end();
    return in;
  }

} // namespace prl

#endif
