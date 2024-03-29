#include <sill/base/variables.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Domain comparisons
  //============================================================================

  template <>
  bool
  includes<domain, finite_domain>(const domain& a, const finite_domain& b) {
    return std::includes(a.begin(), a.end(), b.begin(), b.end());
  }

  template <>
  bool
  includes<domain, vector_domain>(const domain& a, const vector_domain& b) {
    return std::includes(a.begin(), a.end(), b.begin(), b.end());
  }

  template <>
  domain
  set_difference<domain, finite_domain>
  (const domain& a, const finite_domain& b) {
    domain output;
    std::set_difference(a.begin(), a.end(), 
                        b.begin(), b.end(),
                        std::inserter(output, output.begin()));
    return output;
  }

  template <>
  domain
  set_difference<domain, vector_domain>
  (const domain& a, const vector_domain& b) {
    domain output;
    std::set_difference(a.begin(), a.end(), 
                        b.begin(), b.end(),
                        std::inserter(output, output.begin()));
    return output;
  }

  bool set_disjoint(const domain& a, const finite_domain& b) {
    return (intersection_size(a,b) == 0);
  }

  bool set_disjoint(const domain& a, const vector_domain& b) {
    return (intersection_size(a,b) == 0);
  }

  bool set_disjoint(const finite_domain& a, const domain& b) {
    return (intersection_size(a,b) == 0);
  }

  bool set_disjoint(const vector_domain& a, const domain& b) {
    return (intersection_size(a,b) == 0);
  }

  size_t intersection_size(const domain& a, const finite_domain& b) {
    counting_output_iterator counter;
    return std::set_intersection(a.begin(), a.end(), 
                                 b.begin(), b.end(),
                                 counter).count();
  }

  size_t intersection_size(const domain& a, const vector_domain& b) {
    counting_output_iterator counter;
    return std::set_intersection(a.begin(), a.end(), 
                                 b.begin(), b.end(),
                                 counter).count();
  }

  size_t intersection_size(const finite_domain& a, const domain& b) {
    counting_output_iterator counter;
    return std::set_intersection(a.begin(), a.end(), 
                                 b.begin(), b.end(),
                                 counter).count();
  }

  size_t intersection_size(const vector_domain& a, const domain& b) {
    counting_output_iterator counter;
    return std::set_intersection(a.begin(), a.end(), 
                                 b.begin(), b.end(),
                                 counter).count();
  }

  // Domain type conversions
  //============================================================================

  template <>
  void
  convert_domain<domain,finite_domain>
  (const domain& from, finite_domain& to) {
    to.clear();
    foreach(variable* v, from) {
      if (v->type() == variable::FINITE_VARIABLE)
        to.insert((finite_variable*)v);
      else
        assert(false);
    }
  }

  template <>
  void
  convert_domain<domain,vector_domain>
  (const domain& from, vector_domain& to) {
    to.clear();
    foreach(variable* v, from) {
      if (v->type() == variable::VECTOR_VARIABLE)
        to.insert((vector_variable*)v);
      else
        assert(false);
    }
  }

  // Variable vector conversions
  //============================================================================

  finite_var_vector extract_finite_var_vector(const var_vector& vars) {
    finite_var_vector fvars;
    foreach(variable* v, vars) {
      if (v->type() == variable::FINITE_VARIABLE)
        fvars.push_back((finite_variable*)v);
    }
    return fvars;
  }

  vector_var_vector extract_vector_var_vector(const var_vector& vars) {
    vector_var_vector vvars;
    foreach(variable* v, vars) {
      if (v->type() == variable::VECTOR_VARIABLE)
        vvars.push_back((vector_variable*)v);
    }
    return vvars;
  }

  // Vector variable helpers
  //============================================================================

  void
  vector_indices_relative_to_set(const vector_var_vector& vvec,
                                 const vector_domain& vset,
                                 uvec& in_vset_indices,
                                 uvec& notin_vset_indices) {
    size_t in = 0;
    size_t notin = 0;
    foreach(vector_variable* v, vvec) {
      if (vset.count(v))
        in += v->size();
      else
        notin += v->size();
    }
    in_vset_indices.set_size(in);
    notin_vset_indices.set_size(notin);
    in = 0;
    notin = 0;
    size_t n = 0;
    foreach(vector_variable* v, vvec) {
      if (vset.count(v)) {
        for (size_t j = 0; j < v->size(); ++j)
          in_vset_indices[in++] = n++;
      } else {
        for (size_t j = 0; j < v->size(); ++j)
          notin_vset_indices[notin++] = n++;
      }
    }
    assert(n == in_vset_indices.size() + notin_vset_indices.size());
  }

}; // namespace sill

#include <sill/macros_undef.hpp>
