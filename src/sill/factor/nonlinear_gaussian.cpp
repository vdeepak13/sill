#include <sstream>

#include <sill/base/stl_util.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/nonlinear_gaussian.hpp>
#include <sill/factor/operations.hpp>
#include <sill/math/linear_algebra.hpp>

#include <sill/macros_def.hpp>
namespace sill {

  // Constructors and conversion operators
  //============================================================================
  nonlinear_gaussian::nonlinear_gaussian(const vector_var_vector& head,
                                         const vector_var_vector& tail,
                                         const vector_function& fmean,
                                         const gaussian_approximator& approx,
                                         const mat& cov)
    : head(head), tail(tail), fmean_ptr(fmean.clone()), cov(cov),
      approx_ptr(approx.clone()) {
    args.insert(head.begin(), head.end());
    args.insert(tail.begin(), tail.end());
    size_t nh = vector_size(head);
    size_t nt = vector_size(tail);
    // Check the arguments
    assert(args.size() == head.size() + tail.size());
    assert(nh == fmean.size_out());
    assert(nt == fmean.size_in());
    if (cov.size1() == 0 && cov.size2() == 0)
      this->cov = zeros(nh, nh);
    else
      assert(cov.size1() == nh && cov.size2() == nh);
    // Construct the input maps
    fixed_input.resize(nt);
    input_map.resize(nt);
    for(size_t i = 0; i < nt; i++) input_map[i] = i;
    // Install the default approximator
    //approx_ptr.reset(new integration_point_approximator());
  }

  //! Conversion to human-readable representation
  nonlinear_gaussian::operator std::string() const {
    std::ostringstream out; out << *this; return out.str();
  }

  // Queries
  //============================================================================
  canonical_gaussian
  nonlinear_gaussian::approximate(const moment_gaussian& prior) const {
    assert(prior.marginal());
    assert(includes(prior.arguments(), make_domain(tail)));
    moment_gaussian marginal = prior.marginal(make_domain(tail));
    moment_gaussian joint = combine_with(marginal, product_op);
    canonical_gaussian likelihood = canonical_gaussian(joint) / marginal;
    // enforce that likelihood is PSD
    likelihood.enforce_psd(joint.mean(likelihood.argument_list()));
    return likelihood;
  }

  vec nonlinear_gaussian::mean(const vec& input) const {
    vec x = fixed_input;
    for(size_t i = 0; i < input.size(); i++)
      x[input_map[i]] = input[i];
      return fmean()(x);
  }

  // Factor operations
  //============================================================================
  moment_gaussian
  nonlinear_gaussian::combine_with(const moment_gaussian& mg, op_type op) const{
    factor::check_supported(op, product_op);
    assert(mg.marginal());
    vector_domain head_set(head.begin(), head.end());
    vector_domain tail_set(tail.begin(), tail.end());
    assert(tail_set == mg.arguments()); // for now
    assert(set_disjoint(head_set, mg.arguments()));
    return approx()(*this, mg);
  }

  nonlinear_gaussian&
  nonlinear_gaussian::combine_in(const nonlinear_gaussian& other, op_type op) {
    throw std::invalid_argument
      ("nonlinear_gaussian can only be combined with a Gaussian factor");
  }

  nonlinear_gaussian
  nonlinear_gaussian::collapse(const vector_domain& retain, op_type op) const {
    throw std::invalid_argument
      ("nonlinear_gaussian does not support the collapse operation");
  }

  nonlinear_gaussian
  nonlinear_gaussian::restrict(const vector_assignment& a) const {
    vector_domain bound_vars = keys(a);
    if (set_disjoint(bound_vars, args)) return *this;
    // Make a copy
    nonlinear_gaussian result(*this);
    // Update the tail sequence and the input mappings
    result.input_map.clear();
    result.tail.clear();
    size_t k = 0; // the index in the factor input
    for(size_t i = 0; i < tail.size(); i++) {
      vector_variable* v = tail[i];
      if(a.count(v)) { // v is now fixed
        irange ri(input_map[k], input_map[k] + v->size());
        result.fixed_input.set_subvector(ri, safe_get(a, v));
        k += v->size();
      } else {
        result.tail.push_back(v);
        for(size_t j = 0; j < v->size(); j++)
          result.input_map.push_back(input_map[k++]);
      }
    }
    // Update the head assignment and the argument set
    vector_assignment toinsert = map_intersect(a, make_domain(head));
    result.assignment_.insert(toinsert.begin(), toinsert.end());
    result.args = set_difference(args, bound_vars);
    return result;
  }

  nonlinear_gaussian&
  nonlinear_gaussian::subst_args(const vector_var_map& map) {
    args = subst_vars(args, map);
    // replace each element in head by going through the map
    foreach(vector_variable* &a, head) {
      if (map.count(a)) {
        a = safe_get(map, a);
      }
    }
    // replace each element in tail by going through the map
    foreach(vector_variable* &a, tail) {
      if (map.count(a)) {
        a = safe_get(map, a);
      }
    }

    assignment_ = rekey(assignment_, map);
    return *this;
  }

  // Free functions
  //============================================================================
  std::ostream& operator<<(std::ostream& out, const nonlinear_gaussian& cpd) {
    out << "#NG(" << cpd.arguments()
        << "|" << cpd.head_list()
        << "|" << cpd.tail_list() << ")";
    return out;
  }

} // namespace sill

