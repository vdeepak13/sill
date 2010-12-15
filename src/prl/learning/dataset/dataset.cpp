
#include <prl/learning/dataset/dataset.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  // Getters and queries
  //==========================================================================

  record dataset::operator[](size_t i) const {
      record r(finite_numbering_ptr_, vector_numbering_ptr_, dvector);
      load_finite(i, *(r.fin_ptr));
      load_vector(i, *(r.vec_ptr));
      return r;
    }

    record dataset::at(size_t i) const {
      assert(i < nrecords);
      return operator[](i);
    }

    assignment dataset::at_assignment(size_t i) const {
      assert(i < nrecords);
      assignment a;
      load_assignment(i, a);
      return a;
    }

  size_t dataset::finite(size_t i, finite_variable* v) const {
    return finite(i, safe_get(*finite_numbering_ptr_, v));
  }

  /*
  vec dataset::vector(size_t i, vector_variable* v) const {
    assert(v);
    vec val(v->size());
    for (size_t k(0); k < v->size(); ++k)
      val[k] = vector(i, safe_get(*vector_numbering_ptr_, v) + k);
    return val;
  }
  */

  double dataset::vector(size_t i, vector_variable* v, size_t j) const {
    assert(v && (j < v->size()));
    return vector(i, safe_get(*vector_numbering_ptr_, v) + j);
  }

    double dataset::weight(size_t i) const {
      if (weighted)
        return weights_[i];
      else
        return 1;
    }

    double dataset::weight_at(size_t i) const {
      assert(i < size());
      return weight(i);
    }

  void dataset::get_value_matrix(mat& X, const vector_var_vector& vars,
                                 bool add_ones) const {
      assert(includes(vector_vars, vector_domain(vars.begin(), vars.end())));
      size_t vars_size(vector_size(vars));
      if (add_ones) {
        if (X.size1() != nrecords || X.size2() != vars_size + 1)
          X.resize(nrecords, vars_size + 1);
        X.set_col(vars_size, vec(nrecords, 1.));
      } else {
        if (X.size1() != nrecords || X.size2() != vars_size)
          X.resize(nrecords, vars_size);
      }
      for (size_t i(0); i < nrecords; ++i) {
        size_t l2(0); // index into a row in X
        for (size_t j(0); j < vars.size(); ++j) {
          for (size_t k(0); k < vars[j]->size(); ++k) {
            size_t l(safe_get(*vector_numbering_ptr_, vars[j]) + k);
            X(i, l2) = vector(i, l);
            ++l2;
          }
        }
      }
    }

    void dataset::mean(vec& mu, const vector_var_vector& X) const {
      size_t Xsize(0);
      foreach(vector_variable* v, X) {
        if (vector_vars.count(v) == 0)
          throw std::invalid_argument
            ("dataset::covariance() given variable not in dataset");
        Xsize += v->size();
      }
      if (mu.size() != Xsize)
        mu.resize(Xsize);
      mu.zeros();
      if (nrecords == 0)
        return;
      foreach(const record& r, records()) {
        size_t l(0);
        for (size_t j(0); j < X.size(); ++j) {
          size_t ind(safe_get(*vector_numbering_ptr_, X[j]));
          for (size_t k(0); k < X[j]->size(); ++k) {
            mu[l] += r.vector(ind + k);
            ++l;
          }
        }
      }
      mu /= nrecords;
    }

    void dataset::covariance(mat& cov, const vector_var_vector& X) const {
      vec mu;
      mean_covariance(mu, cov, X);
    }

  void dataset::mean_covariance(vec& mu, mat& cov,
                                const vector_var_vector& X) const {
      size_t Xsize(0);
      foreach(vector_variable* v, X) {
        if (vector_vars.count(v) == 0)
          throw std::invalid_argument
            ("dataset::covariance() given variable not in dataset");
        Xsize += v->size();
      }
      if ((cov.size1() != Xsize) || (cov.size2() != Xsize))
        cov.resize(Xsize, Xsize);
      cov.zeros();
      if (nrecords <= 1)
        return;
      mean(mu, X);
      vec tmpvec(Xsize, 0.);
      foreach(const record& r, records()) {
        r.vector_values(tmpvec, X);
        tmpvec -= mu;
        cov += outer_product(tmpvec, tmpvec);
      }
      cov /= (nrecords - 1);
  }

    // Mutating operations
    //==========================================================================

    void dataset::insert(const assignment& a, double w) {
      if (nrecords == capacity()) reserve(2*nrecords);
      size_t i(nrecords);
      ++nrecords;
      set_record(i, a, w);
    }

    void dataset::insert(const std::vector<size_t>& fvals, const vec& vvals,
                         double w) {
      if (nrecords == capacity()) reserve(2*nrecords);
      size_t i(nrecords);
      ++nrecords;
      set_record(i, fvals, vvals, w);
    }

  void dataset::insert_zero_record(double w) {
    if (nrecords == capacity()) reserve(2*nrecords);
    size_t i(nrecords);
    ++nrecords;
    set_record(i, std::vector<size_t>(num_finite(),0), vec(vector_dim(),0), w);
  }

  std::pair<vec, vec> dataset::normalize() {
    vec means(dvector, 0);
    vec std_devs(dvector, 0);
    double total_ds_weight(0);
    for (size_t i = 0; i < nrecords; ++i) {
      for (size_t j = 0; j < dvector; ++j) {
        means[j] += weight(i) * vector(i,j);
        std_devs[j] += weight(i) * vector(i,j) * vector(i,j);
      }
      total_ds_weight += weight(i);
    }
    if (total_ds_weight == 0) {
      means.zeros();
      std_devs.zeros();
      return std::make_pair(means, std_devs);
    }
    means /= total_ds_weight;
    std_devs /= total_ds_weight;
    std_devs = sqrt(std_devs - elem_mult(means, means));
    normalize(means, std_devs);
    return std::make_pair(means, std_devs);
  }

  void dataset::normalize(const vec& means, const vec& std_devs) {
    normalize(means, std_devs, vector_seq);
  }

  std::pair<vec, vec> dataset::normalize(const vector_var_vector& vars) {
    foreach(vector_variable* v, vars)
      assert(vector_vars.count(v) != 0);
    vec means(vector_size(vars), 0);
    vec std_devs(vector_size(vars), 0);
    double total_ds_weight(0);
    ivec vars_inds(vector_indices(vars));
    for (size_t i = 0; i < nrecords; ++i) {
      for (size_t j = 0; j < vars_inds.size(); ++j) {
        size_t j2(vars_inds[j]);
        means[j] += weight(i) * vector(i,j2);
        std_devs[j] += weight(i) * vector(i,j2) * vector(i,j2);
      }
      total_ds_weight += weight(i);
    }
    if (total_ds_weight == 0) {
      means.zeros();
      std_devs.zeros();
      return std::make_pair(means, std_devs);
    }
    means /= total_ds_weight;
    std_devs /= total_ds_weight;
    std_devs = sqrt(std_devs - elem_mult(means, means));
    normalize(means, std_devs, vars);
    return std::make_pair(means, std_devs);
  }

  void dataset::normalize2() {
    normalize2(vector_seq);
  }

    void dataset::randomize() {
      std::time_t time_tmp;
      time(&time_tmp);
      randomize(time_tmp);
    }

    void dataset::make_weighted(double w) {
      assert(weighted == false);
      weights_.resize(capacity());
      for (size_t i = 0; i < nrecords; ++i)
        weights_[i] = w;
      weighted = true;
    }

    void dataset::set_weights(const vec& weights_) {
      assert(weights_.size() == size());
      weighted = true;
      this->weights_ = weights_;
    }

    void dataset::set_weight(size_t i, double weight_) {
      assert(weighted == true);
      assert(i < size());
      weights_[i] = weight_;
    }

  // Free functions
  //==========================================================================

  std::ostream& operator<<(std::ostream& out, const dataset& data) {
    data.print(out);
    return out;
  }

} // namespace prl

#include <prl/macros_undef.hpp>
