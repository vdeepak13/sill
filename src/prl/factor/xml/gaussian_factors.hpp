#ifndef PRL_GAUSSIAN_FACTORS_XML_HPP
#define PRL_GAUSSIAN_FACTORS_XML_HPP

#include <iosfwd>

#include <prl/factor/gaussian_factors.hpp>
#include <prl/archive/xml_iarchive.hpp>
#include <prl/archive/xml_oarchive.hpp>
#include <prl/archive/xml_tag.hpp>

namespace prl {

  template <typename LA>
  const char* xml_tag(canonical_gaussian<LA>*) {
    return "canonical_gaussian";
  }

  template <typename LA>
  xml_oarchive& operator<<(xml_oarchive& out, const canonical_gaussian<LA>& f){
    out.register_variables(f.arguments());
    out.save_begin("canonical_gaussian");
    out.add_attribute("storage", xml_tag((typename LA::value_type*)NULL));

    out << make_nvp("arg_list", f.argument_list());
    out.write_range("inf_matrix", f.inf_matrix().data());
    out.write_range("inf_vector", f.inf_vector());

    out.save_end();
    return out;
  }

  template <typename LA>
  xml_iarchive& operator>>(xml_iarchive& in, canonical_gaussian<LA>& f) {
    in.load_begin("canonical_gaussian");

    // read the arguments
    vector_var_vector args;
    in >> make_nvp("arg_list", args);

    // parse the information matrix and vector
    f = canonical_gaussian<LA>(args, 0);
    in.read_range("inf_matrix", f.inf_matrix().data());
    in.read_range("inf_vector", f.inf_vector());
    
    in.load_end();
    return in;
  }

  template <typename LA>
  const char* xml_tag(moment_gaussian<LA>*) {
    return "moment_gaussian";
  }

  template <typename LA>
  xml_oarchive& operator<<(xml_oarchive& out, const moment_gaussian<LA>& f){
    out.register_variables(f.arguments());
    out.save_begin("moment_gaussian");
    out.add_attribute("storage", xml_tag((typename LA::value_type*)NULL));

    out << make_nvp("head", f.head());
    out << make_nvp("tail", f.tail());
    out.write_range("mean", f.mean());
    out.write_range("cov", f.covariance().data()); 
    out.write_range("coeff", f.coefficients().data()); // row or column-major??

    out.save_end();
    return out;
  }

  template <typename LA>
  xml_iarchive& operator>>(xml_iarchive& in, moment_gaussian<LA>& f) {
    in.load_begin("moment_gaussian");

    // read the arguments
    vector_var_vector head;
    vector_var_vector tail;
    in >> make_nvp("head", head);
    in >> make_nvp("tail", tail);

    // parse the information matrix and vector
    f = moment_gaussian<LA>(head, V(), M(), tail, M());
    in.read_range("mean", f.mean());
    in.read_range("cov", f.covariance().data());
    in.read_range("coeff", f.coefficients().data());
    
    in.load_end();
    return in;
  }

} // namespace prl

#endif
