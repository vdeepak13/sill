#include <stdexcept>

#include <sill/base/stl_util.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/util/operations.hpp>
#include <sill/math/constants.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/serialization/vector.hpp>
#include <sill/range/algorithm.hpp>

#include <sill/macros_def.hpp>
namespace arma {
  std::ostream& operator<<(std::ostream& out, arma::span span) {
    out << span.a << " " << span.b;
    return out;
  }
  

}

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
      assert(cmean.size()==nhead);
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
                  logarithmic<double> likelihood)
    : likelihood(likelihood) {
    initialize(head_list, vector_var_vector());
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

  moment_gaussian::moment_gaussian(const canonical_gaussian& cg)
    : gaussian_base(cg.arguments()), head_list(cg.arg_list),
      coeff(cg.eta.size(),0), likelihood(cg.log_multiplier(), log_tag()) {
    // TO DO: Is likelihood set correctly?
    if (head_list.size() != 0) {
      this->var_span = cg.var_span;
      //size_t n = cg.eta.size();
      //cov.resize(n, n, false);
      bool result = inv(cov, cg.lambda);
      if (!result) {
        throw invalid_operation
          (std::string("The canonical_gaussian does not represent a valid") +
           " marginal distribution.");
      }
      //cmean.resize(n, false);
      cmean = cov * cg.eta;
    }
  }

//   moment_gaussian::operator constant_factor() const {
//     assert(this->arguments().empty());
//     return constant_factor(likelihood);
//   }

//   moment_gaussian::operator std::string() const {
//     std::ostringstream out; out << *this; return out.str();
//   }

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
    var_span.clear();
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
    return (accu(cmean == other.cmean(indh)) == cmean.size() &&
            accu(cov == other.cov(indh, indh)) == cov.n_elem &&
            accu(coeff == other.coeff(indh, indt)) == coeff.n_elem &&
            likelihood == other.likelihood);
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
    assert(y.size() == cmean.size());
    vec yc(y);
    yc -= cmean;
    size_t n = cmean.size();
    double result =
      as_scalar(-0.5*(trans(yc)*(inv(cov)*yc) + n*log(2*pi())+log_det(cov)));
    return logarithmic<double>(result, log_tag()) * likelihood;
  }

  logarithmic<double>
  moment_gaussian::operator()(const vec& y, const vec& x) const {
    using std::log;
    assert(y.size() == cmean.size());
    assert(x.size() == coeff.n_cols);
    vec yc(y);
    yc -= cmean;
    yc -= coeff * x;
    size_t n = cmean.size();
    double result =
      as_scalar(-0.5*(trans(yc)*(inv(cov)*yc) + n*log(2*pi())+log_det(cov)));
    return logarithmic<double>(result, log_tag()) * likelihood;
  }

  double moment_gaussian::log_likelihood(const vector_dataset<>& ds) const {
    canonical_gaussian cg(*this);
    double ll = 0.0;
    foreach (const vector_record<>& r, ds.records(cg.arg_vector())) {
      if (!r.count_missing()) {
        ll += r.weight * log(cg(r.values));
      }
    }
    return ll;
  }

  moment_gaussian&
  moment_gaussian::operator*=(const moment_gaussian& x) {
    return (*this = (*this) * x);
  }

  moment_gaussian&
  moment_gaussian::operator*=(logarithmic<double> x) {
    likelihood *= x;
    return *this;
  }

  moment_gaussian&
  moment_gaussian::operator/=(logarithmic<double> x) {
    likelihood /= x;
    return *this;
  }

  moment_gaussian
  moment_gaussian::marginal(const vector_domain& retain) const {
    // collapse must not eliminate any tail variables
    //assert(vector_domain(tail_list).subset_of(retain));
    assert(includes(retain, vector_domain(tail_list.begin(), tail_list.end())));
    
    vector_domain new_head =
      set_intersect(retain, vector_domain(head_list.begin(), head_list.end()));
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
        new_coeff = columns(coeff, new_tail_ind);
        uvec r_tail_ind;
        this->indices(r_tail, r_tail_ind, true);
        vec r_tail_vals(sill::concat(values(a, r_tail)));
        new_cmean += columns(coeff,r_tail_ind) * r_tail_vals;
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
    bool result = solve(invhh_covhH, cov(ih,ih), cov(ih,iH));
    if (!result) {
      throw invalid_operation
        ("Cholesky decomposition failed in moment_gaussian::collapse");
    }
    double logl = 0;
    logl -= 0.5 * dot(dh, solve(cov(ih,ih), dh));
    logl -= 0.5 * (dh.size() * std::log(2*pi()) + log_det(cov(ih,ih)));
    if (H.size() == 0) {
      return moment_gaussian
        (likelihood * logarithmic<double>(logl, log_tag()));
    } else if (new_tail.size() == 0) {
      return moment_gaussian
        (H,
         new_cmean(iH) + trans(invhh_covhH) * dh,
         cov(iH, iH) - cov(iH,ih) * invhh_covhH,
         likelihood * logarithmic<double>(logl, log_tag()));
    } else {
      return moment_gaussian
        (H,
         new_cmean(iH) + trans(invhh_covhH) * dh,
         cov(iH, iH) - cov(iH,ih) * invhh_covhH,
         new_tail,
         columns(coeff,new_tail_ind),
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
    gaussian_base::subst_args(map);
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

  moment_gaussian
  moment_gaussian::reorder(const vector_var_vector& vars) const {
    assert(vars.size() == arguments().size());

    // head vars must always come first in the vars vector
    vector_var_vector head_vars(vars.begin(), vars.begin() + head().size());
    vector_var_vector tail_vars(vars.begin() + head().size(), vars.end());
    assert(make_domain(head_list) == make_domain(head_vars));
    assert(make_domain(tail_list) == make_domain(tail_vars));

    // get the indices
    uvec head_ind = indices(head_vars);
    uvec tail_ind = indices(tail_vars);
    
    // return the factor
    return moment_gaussian(head_vars,
                           cmean(head_ind),
                           cov(head_ind, head_ind),
                           tail_vars,
                           coeff(head_ind, tail_ind),
                           likelihood);
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
    bool result = solve(cov_ab_cov_b_inv, cov(ib,ib), cov(ib,ia));
    if (!result) {
      throw invalid_operation
        ("Cholesky decomposition failed in moment_gaussian::conditional");
    }
    cov_ab_cov_b_inv = trans(cov_ab_cov_b_inv);
    return moment_gaussian(new_head,
                           cmean(ia) - cov_ab_cov_b_inv * cmean(ib),
                           cov(ia,ia) - cov_ab_cov_b_inv * cov(ib,ia),
                           new_tail, cov_ab_cov_b_inv, 1);
  } // conditional

  bool moment_gaussian::is_conditional(const vector_domain& tail) const {
    return make_domain(tail_list) == tail;
  }

  double moment_gaussian::entropy(double base) const {
    if (!marginal())
      throw std::runtime_error
        ("moment_gaussian::entropy() called for a conditional Gaussian.");
    size_t N(cmean.size());
    return (N + ((N*std::log(2. * pi()) + log_det(cov)) / std::log(base)))/2.;
  }

  double moment_gaussian::entropy() const {
    return entropy(std::exp(1.));
  }

  double moment_gaussian::relative_entropy(const moment_gaussian& q) const {
    assert(arguments() == q.arguments());
    assert(marginal() && q.marginal());
    mat lambdaq = inv(q.covariance(head_list));
    vec mdiff = cmean - q.mean(head_list);
    double d =
      + accu(lambdaq % cov)
      + as_scalar(mdiff.t()*lambdaq*mdiff)
      - mdiff.size()
      - log_det(cov) - log_det(lambdaq);
    return d / 2.0;
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
      return ((log_det(cov(i1,i1)) + log_det(cov(i2,i2)) - log_det(cov)) / 2.);
    } else {
      uvec i12(indices(set_union(d1,d2)));
      return ((log_det(cov(i1,i1)) + log_det(cov(i2,i2)) - log_det(cov(i12,i12)))
              /2.);
    }
  }

  // Private methods
  //==========================================================================

  moment_gaussian
  moment_gaussian::direct_multiplication(const moment_gaussian& x,
                                         const moment_gaussian& y) {
    assert(x.marginal());
    vector_domain args = set_union(x.arguments(), y.arguments());
    moment_gaussian result(args, x.likelihood * y.likelihood);
    uvec xh = result.indices(x.head_list);
    uvec yh = result.indices(y.head_list);
    uvec x_yt  = x.indices(y.tail_list);
    uvec x_all = x.indices(x.head_list);
    result.cmean.elem(xh) = x.cmean;
    result.cmean.elem(yh) = y.coeff * x.cmean(x_yt) + y.cmean;
    mat covyx = y.coeff * x.cov(x_yt, x_all);
    mat covyy = y.coeff * x.cov(x_yt, x_yt) * trans(y.coeff) + y.cov;
    result.cov(xh, xh) = x.cov;
    result.cov(yh, xh) = covyx;
    result.cov(xh, yh) = trans(covyx);
    result.cov(yh, yh) = covyy;
    return result;
  }

  // Free functions
  //============================================================================

  std::ostream& print_vec(std::ostream& out, const arma::vec& vec) {
    out << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0) out << " ";
      out << std::setprecision(5) << vec[i];
    }
    out << "]";
    return out;
  }

  std::ostream& print_mat(std::ostream& out, const arma::mat& mat) {
    out << "[";
    for (size_t i = 0; i < mat.n_rows; ++i) {
      if (i > 0) out << "; ";
      for (size_t j = 0; j < mat.n_cols; ++j) {
        if (j >0) out << " ";
        out << std::setprecision(5) << mat(i,j);
      }
    }
    out << "]";
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const moment_gaussian& mg) {
    out << "#F(MG|" << mg.head() << "|";
    print_vec(out, mg.mean()) << "|";
    print_mat(out, mg.covariance()) << "|";
    if (!mg.marginal()) {
      out << mg.tail() << "|";
      print_mat(out, mg.coefficients()) << "|";
    }
    out << mg.norm_constant() << ")";
    return out;
  }

  double norm_inf(const moment_gaussian& x, const moment_gaussian& y) {
    assert(x.marginal());
    assert(x.head() == y.head());
    double vec_norm =
      norm(x.mean() - y.mean(x.head()), "inf");
    double mat_norm =
      norm(x.covariance() - y.covariance(x.head()), "inf");
    return std::max(vec_norm, mat_norm);
  }

  moment_gaussian
  operator*(const moment_gaussian& x, const moment_gaussian& y) {
    if (x.marginal() &&
        set_disjoint(vector_domain(y.head().begin(), y.head().end()),
                     x.args))
      return moment_gaussian::direct_multiplication(x, y);
    else if (y.marginal() &&
             set_disjoint(vector_domain(x.head().begin(), x.head().end()),
                          y.args))
      return moment_gaussian::direct_multiplication(y, x);
    else
      throw std::invalid_argument("Cannot directly multiply moment Gaussians");
  }

} // namespace sill


