#include <stdexcept>

#include <sill/base/stl_util.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/operations.hpp>
#include <sill/math/constants.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/serialization/vector.hpp>
#include <sill/range/algorithm.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Constructors and conversion operators
  //============================================================================

  void moment_gaussian::initialize(const vector_var_vector& head,
                                   const vector_var_vector& tail){
    // Initialize the indices
    head_list = head;
    tail_list = tail;
    vector_domain head_set(head_list.begin(), head_list.end());
    vector_domain tail_set(tail_list.begin(), tail_list.end());
    assert(set_disjoint(head_set, tail_set));
    args = set_union(head_set, tail_set);

    // Initialize the mapping from variables to index ranges
    compute_indices(head_list);
    compute_indices(tail_list);

    // Compute the expected matrix sizes
    size_t nhead = vector_size(head_list);
    size_t ntail = vector_size(tail_list);

    // Check matrix sizes and initialize empty matrices to defaults
    if (cmean.empty()) {
      cmean = zeros(nhead);
    } else {
      assert(cmean.n_elem==nhead);
    }

    if (cov.n_rows==0 && cov.n_cols==0) {
      cov = eye(nhead, nhead);
    } else {
      assert(cov.n_rows==nhead && cov.n_cols==nhead);
    }

    if (coeff.n_rows==0 && coeff.n_cols==0) {
      coeff = zeros(nhead, ntail);
    } else {
      assert(coeff.n_rows==nhead && coeff.n_cols==ntail);
    }
  }

  moment_gaussian::
  moment_gaussian(const vector_domain& head_list,
                  logarithmic<double> likelihood)
    : likelihood(likelihood) {
    initialize(make_vector(head_list), vector_var_vector());
  }

  moment_gaussian::
  moment_gaussian(const vector_var_vector& head_list,
                  const vector_var_vector& tail_list,
                  logarithmic<double> likelihood)
    : likelihood(likelihood) {
    initialize(head_list, tail_list);
  }

  moment_gaussian::
  moment_gaussian(const vector_var_vector& head_list,
                  const vec& cmean,
                  const mat& cov,
                  logarithmic<double> likelihood)
    : cmean(cmean), cov(cov), likelihood(likelihood) {
    initialize(head_list, vector_var_vector());
  }

  moment_gaussian::
  moment_gaussian(const vector_var_vector& head_list,
                  const vec& cmean,
                  const mat& cov,
                  const vector_var_vector& tail_list,
                  const mat& coeff,
                  logarithmic<double> likelihood)
    : cmean(cmean), cov(cov), coeff(coeff), likelihood(likelihood) {
    initialize(head_list, tail_list);
  }

  moment_gaussian::moment_gaussian(const constant_factor& factor)
    : likelihood(factor.value) {
  }

  moment_gaussian::moment_gaussian(const canonical_gaussian& cg)
    : gaussian_factor(cg.arguments()), head_list(cg.arg_list),
      coeff(cg.eta.n_elem,0), likelihood(cg.log_multiplier(), log_tag()) {
    // TO DO: Is likelihood set correctly?
    if (head_list.size() != 0) {
      this->var_range = cg.var_range;
      //size_t n = cg.eta.n_elem;
      //cov.resize(n, n, false);
      bool result = inv(cg.lambda, cov);
      if (!result) {
        throw invalid_operation
          (std::string("The canonical_gaussian does not represent a valid") +
           " marginal distribution.");
      }
      //cmean.resize(n, false);
      cmean = cov * cg.eta;
    }
  }

  moment_gaussian::operator constant_factor() const {
    assert(this->arguments().empty());
    return constant_factor(likelihood);
  }

  moment_gaussian::operator std::string() const {
    std::ostringstream out; out << *this; return out.str();
  }

  // Serialization
  //==========================================================================

  void moment_gaussian::save(oarchive& ar) const {
    ar << head_list << tail_list << cmean << cov << coeff << likelihood;
  }

  void moment_gaussian::load(iarchive& ar) {
    ar >> head_list >> tail_list >> cmean >> cov >> coeff >> likelihood;
    args.clear();
    args.insert(head_list.begin(), head_list.end());
    args.insert(tail_list.begin(), tail_list.end());
    var_range.clear();
    compute_indices(head_list);
    compute_indices(tail_list);
  }

  // Comparison operators
  //============================================================================

  bool moment_gaussian::operator==(const moment_gaussian& other) const {
    if (arguments() != other.arguments()) return false;
    // FIXME: need to check head and tail separately
    uvec indh = other.indices(head_list);
    uvec indt = other.indices(tail_list);
    return cmean == other.cmean(indh) &&
      cov == other.cov(indh, indh) &&
      coeff == other.coeff(indh, indt) &&
      likelihood == other.likelihood;
  }

  bool moment_gaussian::operator!=(const moment_gaussian& other) const {
    return !operator==(other);
  }

  // Factor operations
  //============================================================================

  logarithmic<double>
  moment_gaussian::operator()(const vector_assignment& a) const {
    vec y(sill::concat(values(a, head_list)));
    if (marginal()) {
      return operator()(y);
    } else {
      vec x(sill::concat(values(a, tail_list)));
      return operator()(y,x);
    }
  }

  logarithmic<double>
  moment_gaussian::operator()(const vec& y) const {
    using std::log;
    if (!marginal())
      throw invalid_operation
        ("moment_gaussian::operator() called on a non-marginal distribution.");
    assert(y.n_elem == cmean.n_elem);
    vec yc(y);
    yc -= cmean;
    size_t n = cmean.n_elem;
    double result = -0.5*(yc*(inv(cov)*yc) + n*log(2*pi())+logdet(cov));
    return logarithmic<double>(result, log_tag()) * likelihood;
  }

  logarithmic<double>
  moment_gaussian::operator()(const vec& y, const vec& x) const {
    using std::log;
    assert(y.n_elem == cmean.n_elem);
    assert(x.n_elem == coeff.n_cols);
    vec yc(y);
    yc -= cmean;
    yc += coeff * x;
    size_t n = cmean.n_elem;
    double result = -0.5*(yc*(inv(cov)*yc) + n*log(2*pi())+logdet(cov));
    return logarithmic<double>(result, log_tag()) * likelihood;
  }

  moment_gaussian&
  moment_gaussian::combine_in(const moment_gaussian& x, op_type op) {
    return (*this = combine(*this, x, op));
  }

  moment_gaussian&
  moment_gaussian::combine_in(const constant_factor& x, op_type op) {
    check_supported(op, combine_ops);
    switch (op) {
    case product_op: likelihood *= x.value; break;
    case divides_op: likelihood /= x.value; break;
    default: check_supported(op, 0);
    }
    return *this;
  }

  moment_gaussian
  moment_gaussian::collapse(op_type op, const vector_domain& retain) const {
    check_supported(op, collapse_ops);
    // collapse must not eliminate any tail variables
    //assert(vector_domain(tail_list).subset_of(retain));
    assert(includes(retain,vector_domain(tail_list.begin(), tail_list.end())));
    
    vector_domain new_head =
      set_intersect(retain,vector_domain(head_list.begin(), head_list.end()));
    vector_var_vector new_head_list = make_vector(new_head);
    uvec ih = indices(new_head_list);
    uvec it = indices(tail_list);

    return moment_gaussian(new_head_list, cmean(ih), cov(ih, ih),
                           tail_list, coeff(ih, it), likelihood);
  }

  moment_gaussian
  moment_gaussian::restrict(const vector_assignment& a) const {

    vector_var_vector H, h; // H = new head_list; h = restricted head_list vars
    foreach(vector_variable* v, head_list) {
      if (a.count(v) == 0)
        H.push_back(v);
      else
        h.push_back(v);
    }
    vector_var_vector new_tail, r_tail; /* new_tail = new tail_list
                                           r_tail = restricted tail_list vars*/
    foreach(vector_variable* v, tail_list) {
      if (a.count(v) == 0)
        new_tail.push_back(v);
      else
        r_tail.push_back(v);
    }

    // If we are not restricting anything, return.
    if (h.size() == 0 && r_tail.size() == 0)
      return *this;

    // Handle the tail first.
    vec new_cmean(cmean);
    mat new_coeff;
    uvec new_tail_ind;
    this->indices(new_tail, new_tail_ind, true);
    if (r_tail.size() == 0) {
      new_coeff = coeff;
    } else {
      if (new_tail.size() == 0) {
        vec v_tail(sill::concat(values(a, tail_list)));
        new_cmean += coeff * v_tail;
      } else {
        new_coeff = coeff.columns(new_tail_ind);
        uvec r_tail_ind;
        this->indices(r_tail, r_tail_ind, true);
        vec r_tail_vals(sill::concat(values(a, r_tail)));
        new_cmean += coeff.columns(r_tail_ind) * r_tail_vals;
      }
    }

    // If no head vars are restricted, return.
    if (h.size() == 0) {
      if (new_tail.size() == 0)
        return moment_gaussian(H, new_cmean, cov, likelihood);
      else
        return moment_gaussian(H, new_cmean, cov, new_tail, new_coeff,
                               likelihood);
    }

    // Now handle the head.
    uvec iH(indices(H)); // new head indices
    uvec ih(indices(h)); // restricted head indices
    vec dh(sill::concat(values(a, h)));
    dh -= new_cmean(ih);
    mat invhh_covhH;
    bool result = ls_solve_chol(cov(ih,ih), cov(ih,iH), invhh_covhH);
    if (!result) {
      throw invalid_operation
        ("Cholesky decomposition failed in moment_gaussian::collapse");
    }
    double logl = 0;
    logl -= 0.5 * dot(dh, ls_solve_chol(cov(ih,ih), dh));
    logl -= 0.5 * (dh.n_elem * std::log(2*pi()) + logdet(cov(ih,ih)));
    if (H.size() == 0) {
      return moment_gaussian
        (likelihood * logarithmic<double>(logl, log_tag()));
    } else if (new_tail.size() == 0) {
      return moment_gaussian
        (H,
         new_cmean(iH) + invhh_covhH.transpose() * dh,
         cov(iH, iH) - cov(iH,ih) * invhh_covhH,
         likelihood * logarithmic<double>(logl, log_tag()));
    } else {
      return moment_gaussian
        (H,
         new_cmean(iH) + invhh_covhH.transpose() * dh,
         cov(iH, iH) - cov(iH,ih) * invhh_covhH,
         new_tail,
         coeff.columns(new_tail_ind),
         likelihood * logarithmic<double>(logl, log_tag()));
    }
  } // restrict(a)

  moment_gaussian&
  moment_gaussian::add_parameters(const moment_gaussian& f, double w) {
    assert(arguments() == f.arguments());
    assert(marginal() && f.marginal());
    uvec ind = indices(f.head_list);
    cmean += w * f.cmean(ind);
    cov += w * f.cov(ind, ind);
    likelihood += w * f.likelihood;
    return *this;
  }

  moment_gaussian&
  moment_gaussian::subst_args(const vector_var_map& map) {
    gaussian_factor::subst_args(map);
    // replace each element in head_list by going through the map
    foreach(vector_variable* &a, head_list) {
      if (map.count(a)) {
        a = safe_get(map, a);
      }
    }
    
    // similarly for tail_list
    foreach(vector_variable* &a, tail_list) {
      if (map.count(a)) {
        a = safe_get(map, a);
      }
    }
    return *this;
  }

  moment_gaussian moment_gaussian::conditional(const vector_domain& B) const {
    if (!marginal())
      throw std::runtime_error
        ("moment_gaussian::conditional() can only be called on marginal distributions.");
    vector_var_vector new_head;
    vector_var_vector new_tail;
    vector_domain all_vars(make_domain<vector_variable>(head_list));
    foreach(vector_variable* v, head_list) {
      if (B.count(v) == 0)
        new_head.push_back(v);
      else
        new_tail.push_back(v);
    }
    if (new_tail.size() != B.size()) {
      throw std::runtime_error
        (std::string("moment_gaussian::conditional()") +
         " given set B with variables not in the factor");
    }
    uvec ia(indices(new_head));
    uvec ib(indices(new_tail));
    mat cov_ab_cov_b_inv;
    bool result = ls_solve_chol(cov(ib,ib), cov(ib,ia), cov_ab_cov_b_inv);
    if (!result) {
      throw invalid_operation
        ("Cholesky decomposition failed in moment_gaussian::conditional");
    }
    cov_ab_cov_b_inv = cov_ab_cov_b_inv.transpose();
    return moment_gaussian(new_head,
                           cmean(ia) - cov_ab_cov_b_inv * cmean(ib),
                           cov(ia,ia) - cov_ab_cov_b_inv * cov(ib,ia),
                           new_tail, cov_ab_cov_b_inv, 1);
  } // conditional

  double moment_gaussian::entropy(double base) const {
    if (!marginal())
      throw std::runtime_error
        ("moment_gaussian::entropy() called for a conditional Gaussian.");
    size_t N(cmean.n_elem);
    return (N + ((N*std::log(2. * pi()) + logdet(cov)) / std::log(base)))/2.;
  }

  double moment_gaussian::entropy() const {
    return entropy(std::exp(1.));
  }

  double moment_gaussian::relative_entropy(const moment_gaussian& q) const {
    assert(false); // not implemented yet
    return 0;
  }

  double moment_gaussian::mutual_information(const vector_domain& d1,
                                             const vector_domain& d2) const {
    // TO DO: Is there a more efficient way to implement this?
    if (!marginal())
      throw std::runtime_error
        ("moment_gaussian::mutual_information() called for a conditional Gaussian.");
    if (!set_disjoint(d1, d2))
      throw std::runtime_error
        ("moment_gaussian::mutual_information() called for non-disjoint sets of variables.");
    if ((!includes(args, d1)) || (!includes(args, d2)))
      throw std::runtime_error
        ("moment_gaussian::mutual_information() called with variables not in the factor arguments.");
    // I(d1; d2) = H(d1) + H(d2) - H(d1,d2)
    uvec i1(indices(d1));
    uvec i2(indices(d2));
    if (args.size() == d1.size() + d2.size()) {
      return ((logdet(cov(i1,i1)) + logdet(cov(i2,i2)) - logdet(cov)) / 2.);
    } else {
      uvec i12(indices(set_union(d1,d2)));
      return ((logdet(cov(i1,i1)) + logdet(cov(i2,i2)) - logdet(cov(i12,i12)))
              /2.);
    }
  }

  // Private methods
  //==========================================================================

  moment_gaussian
  moment_gaussian::direct_combination(const moment_gaussian& x,
                                      const moment_gaussian& y) {
    assert(x.marginal());
    vector_domain args = set_union(x.arguments(), y.arguments());
    moment_gaussian result(args, x.likelihood * y.likelihood);
    uvec xh = result.indices(x.head_list);
    uvec yh = result.indices(y.head_list);
    uvec x_yt  = x.indices(y.tail_list);
    uvec x_all = x.indices(x.head_list);
    result.cmean.set_subvector(xh, x.cmean);
    result.cmean.set_subvector(yh, y.coeff * x.cmean(x_yt) + y.cmean);
    mat covyx = y.coeff * x.cov(x_yt, x_all);
    mat covyy = y.coeff * x.cov(x_yt, x_yt) * y.coeff.transpose() + y.cov;
    result.cov.set_submatrix(xh, xh, x.cov);
    result.cov.set_submatrix(yh, xh, covyx);
    result.cov.set_submatrix(xh, yh, covyx.transpose());
    result.cov.set_submatrix(yh, yh, covyy);
    return result;
  } // direct_combination

  // Free functions
  //============================================================================

  std::ostream& operator<<(std::ostream& out, const moment_gaussian& mg) {
    out << "#F(MG|" << mg.head() << "|\n"
        << "      " << mg.mean() << "|\n"
        << "      " << mg.covariance();
    if (!mg.marginal()) {
      out << "|\n"
          << "      " << mg.tail() << "|\n"
          << "      " << mg.coefficients();
    }
    out << "|\n      " << mg.norm_constant() << ")";
    return out;
  }

  moment_gaussian
  combine(const moment_gaussian& x, const moment_gaussian& y, op_type op) {
    factor::check_supported(op, product_op);
    if (x.marginal() &&
        set_disjoint(vector_domain(y.head().begin(), y.head().end()),
                     x.args))
      return moment_gaussian::direct_combination(x, y);
    else if (y.marginal() &&
             set_disjoint(vector_domain(x.head().begin(), x.head().end()),
                          y.args))
      return moment_gaussian::direct_combination(y, x);
    else
      throw std::invalid_argument("Cannot directly combine moment Gaussians");
  }

} // namespace sill


