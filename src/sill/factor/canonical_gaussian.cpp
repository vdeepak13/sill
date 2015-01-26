#include <algorithm>

#include <sill/base/universe.hpp>
#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/util/operations.hpp>
#include <sill/math/constants.hpp>
#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/serialization/vector.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Constructors, conversion, and initialization
  //============================================================================
  canonical_gaussian::
  canonical_gaussian(const vector_domain& args, logarithmic<double> value)
    : gaussian_base(args), log_mult(log(value)) {
    initialize(make_vector(args));
  }

  canonical_gaussian::
  canonical_gaussian(const vector_var_vector& args, logarithmic<double> value)
    : gaussian_base(args), log_mult(log(value)) {
    initialize(args);
  }

  canonical_gaussian::
  canonical_gaussian(const vector_var_vector& args,
                     const mat& lambda,
                     const vec& eta,
                     double log_mult) {
    reset(args, lambda, eta, log_mult);
  }

  canonical_gaussian::canonical_gaussian(const moment_gaussian& mg)
    : gaussian_base(mg.arguments()) {
    size_t nhead = mg.size_head();
    size_t ntail = mg.size_tail();
    size_t n = nhead + ntail;
    lambda.set_size(n, n);
    eta.set_size(n);

    // Initialize the argument list
    arg_list = concat(mg.head_list, mg.tail_list);

    // Initialize the argument map
    compute_indices(arg_list);

    // Initialize the matrices
    span ih(0, nhead-1);
    span it(nhead, n-1);
    mat invcov = inv(mg.cov);

    if (nhead > 0) {
      lambda(ih, ih) = invcov;
      eta.subvec(ih) = invcov * mg.cmean;
    }

    if (ntail > 0) {
      mat Atinvcov = trans(mg.coeff) * invcov;
      lambda(it, it) = Atinvcov*mg.coeff;
      lambda(ih, it) = -trans(Atinvcov);
      lambda(it, ih) = -Atinvcov;
      eta.subvec(it) = -Atinvcov * mg.cmean;
    }

    log_mult = mg.likelihood.log_value()
      - .5 * (mg.cmean.size() * std::log(2*pi<double>()) + log_det(mg.cov)
              + as_scalar(trans(mg.cmean) * invcov * mg.cmean));
  }

  void canonical_gaussian::initialize(const vector_var_vector& args) {
    arg_list = args;
    compute_indices(args);
    size_t n = vector_size(arg_list);
    lambda = zeros(n, n);
    eta = zeros(n);
  }

  void canonical_gaussian::reset(const vector_var_vector& args,
                                 const mat& lambda,
                                 const vec& eta,
                                 double log_mult) {
    this->args.clear();
    this->args.insert(args.begin(), args.end());
    this->arg_list = args;
    this->var_span.clear();
    compute_indices(args);
    
    this->lambda = lambda;
    this->eta = eta;
    this->log_mult = log_mult;
    size_t n = vector_size(arg_list);
    assert(lambda.n_rows == n && lambda.n_cols == n);
    assert(eta.n_rows == n);
  }

  // Serialization
  //============================================================================

  void canonical_gaussian::save(oarchive& ar) const {
    ar << arg_list << lambda << eta << log_mult;
  }

  void canonical_gaussian::load(iarchive& ar) {
    ar >> arg_list >> lambda >> eta >> log_mult;
    args = vector_domain(arg_list.begin(), arg_list.end());
    var_span.clear();
    compute_indices(arg_list);
  }

  // Accessors
  //==========================================================================

  const vector_var_vector& canonical_gaussian::arg_vector() const {
    return arg_list;
  }

  size_t canonical_gaussian::size() const {
    return eta.size();
  }

  const mat& canonical_gaussian::inf_matrix() const {
    return lambda;
  }
    
  mat& canonical_gaussian::inf_matrix() {
    return lambda;
  }

  const vec& canonical_gaussian::inf_vector() const { 
    return eta;
  }

  vec& canonical_gaussian::inf_vector() {
    return eta;
  }

  double canonical_gaussian::log_multiplier() const {
    return log_mult;
  }

  double& canonical_gaussian::log_multiplier() {
    return log_mult;
  }

  mat canonical_gaussian::inf_matrix(const vector_var_vector& args) const {
    uvec ind(indices(args));
    return lambda(ind, ind);
  }

  vec canonical_gaussian::inf_vector(const vector_var_vector& args) const {
    uvec ind(this->indices(args));
    return eta(ind);
  }

  // Comparison operators
  //============================================================================
  bool canonical_gaussian::operator==(const canonical_gaussian& other) const {
    return (arguments() == other.arguments() &&
            equal(inf_vector(), other.inf_vector(arg_list)) &&
            equal(inf_matrix(), other.inf_matrix(arg_list)));
  }

  bool canonical_gaussian::operator!=(const canonical_gaussian& other) const {
    return !operator==(other);
  }

  bool canonical_gaussian::operator<(const canonical_gaussian& other) const {
    if (this->arguments() < other.arguments()) return true;

    if (this->arguments() == other.arguments()) {
      vec other_vector = other.inf_vector(arg_list);
      if (boost::lexicographical_compare(inf_vector(), other_vector))
        return true; // inf_vector() < other_vector

      if (equal(inf_vector(), other_vector)) {
        mat other_matrix = other.inf_matrix(arg_list);
        for(size_t i = 0; i<lambda.n_rows; i++) {
          for(size_t j = 0; j<lambda.n_cols; j++) {
            if (lambda(i,j) != other_matrix(i,j))
              return lambda(i,j) < other_matrix(i,j);
          }
        }
        return false; // (*this) == other
      }
    }
    return false;
  }

  // Factor operations
  //==========================================================================
  logarithmic<double>
  canonical_gaussian::operator()(const vector_assignment& a) const {
    return logarithmic<double>(logv(a), log_tag());
  }

  logarithmic<double>
  canonical_gaussian::operator()(const record_type& r) const {
    return logarithmic<double>(logv(r), log_tag());
  }
  
  logarithmic<double>
  canonical_gaussian::operator()(const vec& v) const {
    double log = - 0.5*dot(v, lambda*v) + dot(v, eta) + log_mult;
    return logarithmic<double>(log, log_tag());
  }

  double canonical_gaussian::logv(const vector_assignment& a) const {
    if (eta.is_empty())
      return log_mult;
    vec v = sill::concat(values(a, arg_list));
    // will assertion if a does not cover the arguments of this
    return - 0.5*dot(v, lambda*v) + dot(v, eta) + log_mult;
  }

  double canonical_gaussian::logv(const record_type& r) const {
    if (eta.is_empty())
      return log_mult;
    vec v = zeros(eta.size());
    r.vector_values(v, arg_list);
    // will assertion if a does not cover the arguments of this
    return - 0.5*as_scalar(trans(v) * lambda * v) + dot(v, eta) + log_mult;
  }

  canonical_gaussian& canonical_gaussian::operator*=(const canonical_gaussian& x) {
    return combine_in(x, 1.0);
  }

  canonical_gaussian& canonical_gaussian::operator/=(const canonical_gaussian& x) {
    return combine_in(x, -1.0);
  }

  canonical_gaussian& canonical_gaussian::operator*=(logarithmic<double> val) {
    log_mult += log(val);
    return *this;
  }

  canonical_gaussian& canonical_gaussian::operator/=(logarithmic<double> val) {
    log_mult -= log(val);
    return *this;
  }

  canonical_gaussian
  canonical_gaussian::marginal(const vector_domain& retain) const {
    canonical_gaussian cg;
    marginal(retain, cg);
    return cg;
  }

  void
  canonical_gaussian::marginal(const vector_domain& retain,
                               canonical_gaussian& cg) const {
    marginal(retain, true, cg);
  }

  void
  canonical_gaussian::marginal_unnormalized(const vector_domain& retain,
                                            canonical_gaussian& cg) const {
    marginal(retain, false, cg);
  }

  canonical_gaussian
  canonical_gaussian::maximum(const vector_domain& retain) const {
    vector_assignment a;
    vector_assignment means(arg_max()); // This could be more efficient.
    foreach(vector_variable* v, arg_list)
      if (retain.count(v) == 0)
        a[v] = means[v];
    return restrict(a);
  }

  vector_assignment canonical_gaussian::arg_max() const {
    mat cov;
    if (lambda.size() == 0)
      return vector_assignment();
    bool result = inv(cov, lambda);
    if (!result) {
      std::cerr << "In canonical_gaussian::arg_max, could not invert lambda =\n"
                << lambda << std::endl;
      throw invalid_operation("The canonical_gaussian does not represent a valid marginal distribution.");
    }
    vec mu(cov * eta);
    return make_assignment(arg_list, mu);
  }

  canonical_gaussian
  canonical_gaussian::restrict(const vector_assignment& a) const {
    // Determine the retained (x) and the restricted variables (y)
    vector_var_vector x, y;
    foreach(vector_variable* v, arg_list) {
      if (a.count(v) == 0)
        x.push_back(v);
      else
        y.push_back(v);
    }
    // If the arguments of x are disjoint from the bound variables,
    // we can simply return a copy of the factor
    if (y.empty()) return *this;

    uvec ix = indices(x);
    uvec iy = indices(y);
    vec vy = sill::concat(values(a, y));
    assert(vy.size() == iy.size());

    double lm = log_mult + dot(eta(iy), vy)
      - 0.5 * as_scalar(trans(vy) * lambda(iy,iy) * vy);
    if (x.empty()) {
      return canonical_gaussian(logarithmic<double>(lm, log_tag()));
    } else {
      return canonical_gaussian(x, lambda(ix,ix), eta(ix) - lambda(ix,iy)*vy, lm);
    }
  }

  void canonical_gaussian::
  restrict(const record_type& r, const vector_domain& r_vars,
           canonical_gaussian& f) const {
    this->restrict(r, r_vars, false, f);
  }

  void canonical_gaussian::
  restrict(const record_type& r, const vector_domain& r_vars,
           bool strict, canonical_gaussian& f) const {
    // Determine the retained (x) and the restricted variables (y)
    vector_var_vector x, y;
    foreach(vector_variable* v, arg_list) {
      if (r_vars.count(v) == 0) {
        x.push_back(v);
      } else {
        if (!r.has_variable(v)) {
          if (strict) {
            throw std::invalid_argument
              (std::string("canonical_gaussian::restrict(r,r_vars,strict,f)") +
               " was given strict=true, but intersect(f.arguments(), r_vars)" +
               " contained a variable which did not appear in keys(r).");
          }
          x.push_back(v);
        } else {
          y.push_back(v);
        }
      }
    }

    // If the arguments of x are disjoint from the bound variables,
    // we can simply return a copy of the factor
    if (y.empty()) {
      f = *this;
      return;
    }

    uvec ix = indices(x);
    uvec iy = indices(y);
    vec vy;
    r.vector_values(vy, y);
    assert(vy.size() == iy.size());

    double lm = log_mult + dot(eta(iy), vy)
      - 0.5*as_scalar(trans(vy) * lambda(iy,iy) * vy);

    if (x.empty()) {
      f = canonical_gaussian(logarithmic<double>(lm, log_tag()));
    } else {
      f = canonical_gaussian(x, lambda(ix,ix), eta(ix) - lambda(ix,iy)*vy, lm);
    }
  } // restrict(f, r, r_vars, strict)

  canonical_gaussian&
  canonical_gaussian::subst_args(const vector_var_map& map) {
    gaussian_base::subst_args(map);

    foreach(vector_variable* &a, arg_list) {
      if (map.count(a)) {
        a = safe_get(map, a);
      }
    }
    return *this;
  }

  canonical_gaussian
  canonical_gaussian::reorder(const vector_var_vector& args) const {
    assert(args.size() == arguments().size());
    return canonical_gaussian(args,
                              inf_matrix(args),
                              inf_vector(args),
                              log_mult);
  }

  canonical_gaussian
  canonical_gaussian::conditional(const vector_domain& B) const {
    assert(includes(arguments(), B));
    canonical_gaussian PB(marginal(B));
    return (*this) / PB;
  }

  bool canonical_gaussian::is_normalizable() const {
    if (lambda.size() == 0)
      return true;
    mat lambda_inv;
    bool result = inv(lambda_inv, lambda);
    return result;
  }

  double canonical_gaussian::norm_constant() const {
    return std::exp(log_norm_constant());
  }

  double canonical_gaussian::log_norm_constant() const {
    mat lambda_inv;
    if (lambda.is_empty())
      return log_mult;
    try {
      lambda_inv = inv(lambda);
    } catch(std::runtime_error& e) {
      if (lambda.size() <= 16)
        std::cerr << "Lambda:\n" << lambda << std::endl;
      throw invalid_operation("Inversion of lambda matrix failed in canonical_gaussian::log_norm_constant.");
    }
    return log_mult + (0.5 * (eta.size() * std::log(2*pi<double>())
                              - log_det(lambda)
                              + as_scalar(trans(eta) * lambda_inv * eta)));
  }

  canonical_gaussian& canonical_gaussian::normalize() {
    log_mult -= log_norm_constant();
    return *this;
  }

  double canonical_gaussian::entropy(double base) const {
    size_t N = eta.size();
    return (N + ((N*std::log(2.0 * pi<double>()) - log_det(lambda)) / std::log(base)))/2.0;
  }

  double canonical_gaussian::entropy() const {
    return entropy(e<double>());
  }

  double canonical_gaussian::relative_entropy(const canonical_gaussian& q) const
  {
    assert(false); // not implemented yet
    return 0;
  }

  double canonical_gaussian::mutual_information(const vector_domain& d1,
                                                const vector_domain& d2) const {
    // TO DO: Make this more efficient.
    if (!set_disjoint(d1, d2))
      throw std::runtime_error
        ("canonical_gaussian::mutual_information() called for non-disjoint sets of variables.");
    if ((!includes(args, d1)) || (!includes(args, d2)))
      throw std::runtime_error
        ("canonical_gaussian::mutual_information() called with variables not in the factor arguments.");
    // I(d1; d2) = H(d1) + H(d2) - H(d1,d2)
    canonical_gaussian cg1(marginal(d1));
    canonical_gaussian cg2(marginal(d2));
    return cg1.entropy() + cg2.entropy() - entropy();
  }

  // Factor operations: combining factors
  //==========================================================================

  canonical_gaussian&
  canonical_gaussian::combine_in(const canonical_gaussian& x, double sign) {
    if (!includes(arguments(), x.arguments())) {
//       size_t n(eta.size());
//       for (vector_domain::const_iterator it(x.args.begin());
//            it != x.args.end(); ++it) {
//         vector_variable* v = *it;
//         if (args.insert(v).second) { // then variable was not yet in args
//           var_span[v] = span(n, n + v->size() - 1);
//           n += v->size();
//           arg_list.push_back(v);
//         }
//       }
//       lambda.resize(n, n, true);
//       eta.resize(n, true);
      *this = combine(*this, x, sign);
      return *this;
    }
    uvec indx;
    indices(x.arg_list, indx);
    if (sign == 1) {
      lambda(indx, indx) += x.lambda;
      eta(indx) += x.eta;
      log_mult += x.log_mult;
    } else {
      lambda(indx, indx) -= x.lambda;
      eta(indx) -= x.eta;
      log_mult -= x.log_mult;
    }
    return *this;
  }

  // Other operations
  //==========================================================================
  bool canonical_gaussian::enforce_psd(const vec& mean) {
    assert(mean.size() == size());
    vec d;
    mat v;
    bool result = eig_sym(d, v, lambda);
    if (!result) {
      using namespace std;
      cerr << lambda << endl;
      assert(result);
    }
    double mine = min(d);
    if (mine < 0) {
//       std::cerr << "Information matrix not PSD (min eigenvalue " << mine
//                 << "); adjusting." << std::endl;
      for(size_t i = 0; i < d.size(); i++) 
        d[i] = std::max(0.0, d[i]);
      mat new_lambda = v * diagmat(d) * trans(v);
      eta += (new_lambda - lambda) * mean;
      lambda = new_lambda;
      return false;
    } else {
      return true;
    }
  }

  void canonical_gaussian::marginal(const vector_domain& retain,
                                    bool renormalize,
                                    canonical_gaussian& cg) const {
    if (retain.empty()) {
      cg.var_span.clear();
      cg.args.clear();
      cg.arg_list.clear();
      cg.lambda.reset();
      cg.eta.reset();
      if (renormalize)
        cg.log_mult = log_norm_constant();
      return;
    }
    vector_var_vector x, y; // retained, marginalized out
    foreach(vector_variable* v, arg_list) {
      if (retain.count(v) == 0)
        y.push_back(v);
      else
        x.push_back(v);
    }
    if (y.empty()) {
      if (cg.arg_list == x) {
        cg.lambda = lambda;
        cg.eta = eta;
        cg.log_mult = log_mult;
      } else {
        cg = *this;
      }
    } else {
      uvec ix = indices(x);
      uvec iy = indices(y);
      mat invyy_lamyx;
      //bool info = ls_solve_chol(lambda(iy,iy), lambda(iy,ix), invyy_lamyx);
      // Armadillo does not yet suppport solve using CHolesky decomposition
      // try solving via LU factorization
      // Note: ls_solve_chol failed on some symmetric matrices with
      //       positive determinants, so it's possible the IT++
      //       implementation is buggy.
      //       (ls_solve has worked in these cases so far.)
      bool info = solve(invyy_lamyx, lambda(iy,iy), lambda(iy,ix));
      // 
      if (!info) {
        if (iy.size() * ix.size() < 16 &&
            iy.size() * iy.size() < 16)
          std::cerr << "Lambda(iy,iy):\n" << lambda(iy,iy) << "\n"
                    << "Lambda(iy,ix):\n" << lambda(iy,ix) << std::endl;
        throw invalid_operation
          (std::string("canonical_gaussian::collapse:") +
           " LU factorization failed.");
      }
      if (cg.arg_list == x) {
        cg.lambda = lambda(ix,ix) - lambda(ix,iy) * invyy_lamyx;
        cg.eta = eta(ix) - trans(invyy_lamyx) * eta(iy);
        cg.log_mult = 0;
      } else {
        cg.reset(x,
                 lambda(ix,ix) - lambda(ix,iy) * invyy_lamyx,
                 eta(ix) - trans(invyy_lamyx) * eta(iy));
      }
      if (renormalize) {
        cg.log_mult = log_norm_constant() - cg.log_norm_constant();
      }
    }
  } // marginal

  // Free functions
  //==========================================================================
  inline canonical_gaussian combine(const canonical_gaussian& x,
                                    const canonical_gaussian& y,
                                    double sign) {
    vector_domain args = set_union(x.arguments(), y.arguments());
    canonical_gaussian result(args, 0);
    uvec indx = result.indices(x.arg_list);
    uvec indy = result.indices(y.arg_list);

    result.lambda(indx, indx)  = x.lambda;
    result.lambda(indy, indy) += sign * y.lambda;
    result.eta(indx)  = x.eta;
    result.eta(indy) += sign * y.eta;
    result.log_mult = x.log_mult + sign * y.log_mult;

    return result;
  }

  canonical_gaussian operator*(const canonical_gaussian& x,
                               const canonical_gaussian& y) {
    return combine(x, y, +1.0);
  }

  canonical_gaussian operator/(const canonical_gaussian& x,
                               const canonical_gaussian& y) {
    return combine(x, y, -1.0);
  }

  double norm_inf(const canonical_gaussian& x, const canonical_gaussian& y) {
    assert(x.arguments() == y.arguments());
    double vec_norm =
      norm(x.inf_vector() - y.inf_vector(x.arg_vector()), "inf");
    double mat_norm =
      norm(x.inf_matrix() - y.inf_matrix(x.arg_vector()), "inf");
    return std::max(vec_norm, mat_norm);
  }

  canonical_gaussian pow(const canonical_gaussian& cg, double a) {
    return canonical_gaussian(cg.arg_vector(),
                              cg.inf_matrix() * a,
                              cg.inf_vector() * a,
                              cg.log_multiplier() * a);
  }

  vector_assignment arg_max(const canonical_gaussian& cg) {
    return cg.arg_max();
  }

  canonical_gaussian weighted_update(const canonical_gaussian& f1,
                                     const canonical_gaussian& f2,
                                     double a) {
    if (a == 1)
      return f2;
    else if (a == 0)
      return f1;
    else
      return pow(f1, 1-a) * pow(f2, a);
  }

  canonical_gaussian invert(const canonical_gaussian& cg) {
    return canonical_gaussian(cg.arg_vector(),
                              -cg.inf_matrix(),
                              -cg.inf_vector(),
                              -cg.log_multiplier());
  }

  std::ostream& operator<<(std::ostream& out, const canonical_gaussian& cg) {
    out << "#F(CG|" << cg.arg_vector()
        << "|" << cg.inf_matrix()
        << "|" << cg.inf_vector()
        << "|" << cg.log_multiplier() << ")";
    return out;
  }

} // namespace sill

