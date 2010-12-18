
#include <iostream>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/data_loader.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  namespace data_loader {

    // Free functions
    //==========================================================================

  boost::shared_ptr<sill::oracle>
  load_oracle(sill::universe& u, std::string ds_name, double random_seed) {
    using namespace sill;
    boost::shared_ptr<sill::oracle> oracle_ptr;
    if (ds_name == "knorm") {
      size_t nfeatures = 20;
      size_t k = 2;
      vector_var_vector var_order;
      std::vector<variable::variable_typenames> var_type_order
        (nfeatures, variable::VECTOR_VARIABLE);
      var_type_order.push_back(variable::FINITE_VARIABLE);
      for (size_t j = 0; j < nfeatures; j++)
        var_order.push_back(u.new_vector_variable(1));
      syn_oracle_knorm::parameters params;
      params.random_seed = random_seed;
      oracle_ptr.reset(new syn_oracle_knorm(var_order, u.new_finite_variable(k),
                                            var_type_order, params));
    } else if (ds_name == "majority") {
      size_t nfeatures = 100;
      finite_var_vector var_order;
      for (size_t j = 0; j <= nfeatures; j++)
        var_order.push_back(u.new_finite_variable(2));
      syn_oracle_majority::parameters params;
      params.random_seed = random_seed;
      oracle_ptr.reset(new syn_oracle_majority(var_order, params));
    } else {
      symbolic::parameters params(load_symbolic_summary(ds_name, u));
      oracle_ptr.reset(new symbolic_oracle(params));
    }
    return oracle_ptr;
  }

  boost::shared_ptr<sill::oracle>
  load_oracle(const sill::datasource_info_type& info,
                           std::string ds_name, double random_seed) {
    using namespace sill;
    boost::shared_ptr<sill::oracle> oracle_ptr;
    if (ds_name == "knorm") {
      syn_oracle_knorm::parameters params;
      params.random_seed = random_seed;
      oracle_ptr.reset(new syn_oracle_knorm(info.vector_seq,
                                            info.finite_seq.front(),
                                            info.var_type_order, params));
    } else if (ds_name == "majority") {
      syn_oracle_majority::parameters params;
      params.random_seed = random_seed;
      oracle_ptr.reset(new syn_oracle_majority(info.finite_seq, params));
    } else {
      symbolic::parameters params(load_symbolic_summary(ds_name, info));
      oracle_ptr.reset(new symbolic_oracle(params));
    }
    return oracle_ptr;
  }

  boost::shared_ptr<symbolic_oracle>
  load_symbolic_oracle(const std::string& filename, universe& u,
                       size_t record_limit, bool auto_reset) {
    symbolic::parameters sym_params(load_symbolic_summary(filename, u));
    symbolic_oracle::parameters params;
    params.record_limit = record_limit;
    params.auto_reset = auto_reset;
    return boost::shared_ptr<symbolic_oracle>
      (new symbolic_oracle(sym_params, params));
  }

  boost::shared_ptr<symbolic_oracle>
  load_symbolic_oracle
  (const std::string& filename, const datasource_info_type& info,
   size_t record_limit, bool auto_reset) {
    symbolic::parameters sym_params(load_symbolic_summary(filename,info));
    symbolic_oracle::parameters params;
    params.record_limit = record_limit;
    params.auto_reset = auto_reset;
    return boost::shared_ptr<symbolic_oracle>
      (new symbolic_oracle(sym_params, params));
  }

  // Free functions: Utilities
  //==========================================================================

  template <>
  void save_variables<finite_variable>
  (std::ostream& out, const finite_domain& vars, const datasource& ds) {
    out << "f[";
    foreach(finite_variable* v, vars)
      out << ds.record_index(v) << " ";
    out << "]";
  }

  template <>
  void save_variables<vector_variable>
  (std::ostream& out, const vector_domain& vars, const datasource& ds) {
    out << "v[";
    foreach(vector_variable* v, vars)
      out << ds.record_index(v) << " ";
    out << "]";
  }

  template <>
  void save_variables<variable>
  (std::ostream& out, const domain& vars, const datasource& ds) {
    out << "fv[";
    foreach(variable* v, vars)
      out << ds.var_order_index(v) << " ";
    out << "]";
  }

  template <>
  void load_variables<finite_variable>
  (std::istream& in, finite_domain& vars, const datasource& ds) {
    assert(in.peek() == 'f');
    in.ignore(1);
    assert(in.peek() == '[');
    in.ignore(1);
    vars.clear();
    size_t tmpsize;
    const finite_var_vector& finite_list = ds.finite_list();
    while (in.peek() != ']') {
      if (!(in >> tmpsize))
        assert(false);
      assert(tmpsize < finite_list.size());
      vars.insert(ds.finite_list()[tmpsize]);
      if (in.peek() == ' ')
        in.ignore(1);
    }
    in.ignore(1);
  }

  template <>
  void load_variables<vector_variable>
  (std::istream& in, vector_domain& vars, const datasource& ds) {
    assert(in.peek() == 'v');
    in.ignore(1);
    assert(in.peek() == '[');
    in.ignore(1);
    vars.clear();
    size_t tmpsize;
    const vector_var_vector& vector_list = ds.vector_list();
    while (in.peek() != ']') {
      if (!(in >> tmpsize))
        assert(false);
      assert(tmpsize < vector_list.size());
      vars.insert(ds.vector_list()[tmpsize]);
      if (in.peek() == ' ')
        in.ignore(1);
    }
    in.ignore(1);
  }

  template <>
  void load_variables<variable>
  (std::istream& in, domain& vars, const datasource& ds) {
    assert(in.peek() == 'f');
    in.ignore(1);
    assert(in.peek() == 'v');
    in.ignore(1);
    assert(in.peek() == '[');
    in.ignore(1);
    vars.clear();
    size_t tmpsize;
    var_vector variable_list(ds.var_order());
    while (in.peek() != ']') {
      if (!(in >> tmpsize))
        assert(false);
      assert(tmpsize < variable_list.size());
      vars.insert(variable_list[tmpsize]);
      if (in.peek() == ' ')
        in.ignore(1);
    }
    in.ignore(1);
  }

  } // namespace data_loader

} // namespace sill

#include <sill/macros_undef.hpp>
