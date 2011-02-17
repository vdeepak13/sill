#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/math/norms.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Protected methods required by record
  //==========================================================================

  void vector_dataset::load_assignment(size_t i, sill::assignment& a) const {
    assert(i < nrecords);
    convert_finite_record2assignment(finite_data[i], a.finite());
    convert_vector_record2assignment(vector_data[i], a.vector());
  }

  void vector_dataset::load_record(size_t i, record& r) const {
    if (r.fin_own)
      r.fin_ptr->operator=(finite_data[i]);
    else
      r.fin_ptr = &(finite_data[i]);
    if (r.vec_own)
      r.vec_ptr->operator=(vector_data[i]);
    else
      r.vec_ptr = &(vector_data[i]);
  }

  void vector_dataset::load_assignment_pointer(size_t i, assignment** a) const {
    assert(false);
  }

  // Protected methods
  //==========================================================================

  void vector_dataset::init(size_t nreserved) {
    //      assert(nreserved > 0);
    finite_data.resize(nreserved);
    for (size_t i = 0; i < nreserved; ++i)
      finite_data[i].resize(num_finite());
    vector_data.resize(nreserved);
    for (size_t i = 0; i < nreserved; ++i)
      vector_data[i].resize(dvector);
  }

  // Constructors
  //==========================================================================

  void vector_dataset::save(oarchive& a) const {
    base::save(a);
    a << finite_data << vector_data;
  }

  void vector_dataset::load(iarchive& a) {
    base::load(a);
    a >> finite_data >> vector_data;
  }

  // Getters and helpers
  //==========================================================================

  std::pair<vector_dataset::record_iterator, vector_dataset::record_iterator>
  vector_dataset::records() const {
    if (nrecords > 0)
      return std::make_pair
        (make_record_iterator(0, &(finite_data[0]), &(vector_data[0])),
         make_record_iterator(nrecords));
    else
      return std::make_pair
        (make_record_iterator(0), make_record_iterator(nrecords));
  }

  vector_dataset::record_iterator vector_dataset::begin() const {
    if (nrecords > 0)
      return make_record_iterator(0, &(finite_data[0]), &(vector_data[0]));
    else
      return make_record_iterator(0);
  }

  // Mutating operations
  //==========================================================================

  void vector_dataset::reserve(size_t n) {
    if (n > capacity()) {
      size_t n_previous = finite_data.size();
      finite_data.resize(n);
      for (size_t i = n_previous; i < n; ++i)
        finite_data[i].resize(num_finite());
      vector_data.resize(n);
      for (size_t i = n_previous; i < n; ++i)
        vector_data[i].resize(dvector);
      if (weighted)
        weights_.resize(n, true);
    }
  }

  void vector_dataset::set_record(size_t i, const assignment& a, double w) {
    assert(i < nrecords);
    convert_finite_assignment2record(a.finite(), finite_data[i]);
    convert_vector_assignment2record(a.vector(), vector_data[i]);
    if (weighted) {
      weights_[i] = w;
    } else if (w != 1) {
      std::cerr << "vector_dataset::set_record() called with weight w != 1"
                << " on an unweighted dataset." << std::endl;
      assert(false);
    }
  }

  void vector_dataset::set_record(size_t i, const std::vector<size_t>& fvals,
                                  const vec& vvals, double w) {
    assert(i < nrecords);
    finite_data[i] = fvals;
    vector_data[i] = vvals;
    if (weighted) {
      weights_[i] = w;
    } else if (w != 1) {
      std::cerr << "vector_dataset::set_record() called with weight w != 1"
                << " on an unweighted dataset." << std::endl;
      assert(false);
    }
  }

  std::pair<vec, vec> vector_dataset::normalize() {
    vec means(dvector,0);
    vec std_devs(dvector,0);
    double total_ds_weight(0);
    vec tmpvec(dvector,0);
    for (size_t i = 0; i < nrecords; ++i) {
      means += weight(i) * vector_data[i];
      tmpvec = vector_data[i];
      tmpvec *= tmpvec;
      std_devs += weight(i) * tmpvec;
      total_ds_weight += weight(i);
    }
    means /= total_ds_weight;
    std_devs /= total_ds_weight;
    for (size_t j = 0; j < dvector; ++j)
      std_devs[j] = sqrt(std_devs[j] - means[j] * means[j]);
    normalize(means, std_devs);
    return std::make_pair(means, std_devs);
  }

  void vector_dataset::normalize(const vec& means,
                                 const vec& std_devs) {
    assert(means.size() == dvector);
    assert(std_devs.size() == dvector);
    vec stddevs(std_devs);
    for (size_t j(0); j < dvector; ++j) {
      if (stddevs[j] < 0)
        assert(false);
      if (stddevs[j] == 0)
        stddevs[j] = 1;
    }
    for (size_t i(0); i < nrecords; ++i) {
      vector_data[i] -= means;
      vector_data[i] /= stddevs;
    }
  }

  void vector_dataset::normalize(const vec& means, const vec& std_devs,
                                 const vector_var_vector& vars) {
    foreach(vector_variable* v, vars)
      assert(vector_vars.count(v) != 0);
    ivec vars_inds(vector_indices(vars));
    assert(means.size() == vars_inds.size());
    assert(std_devs.size() == vars_inds.size());
    vec stddevs(std_devs);
    for (size_t j(0); j < stddevs.size(); ++j) {
      if (stddevs[j] < 0)
        assert(false);
      if (stddevs[j] == 0)
        stddevs[j] = 1;
    }
    for (size_t i(0); i < nrecords; ++i) {
      for (size_t j(0); j < stddevs.size(); ++j) {
        size_t j2(vars_inds[j]);
        vector_data[i](j2) -= means(j);
        vector_data[i](j2) /= stddevs(j);
      }
    }
  }

  void vector_dataset::normalize2(const vector_var_vector& vars) {
    foreach(vector_variable* v, vars)
      assert(vector_vars.count(v) != 0);
    ivec vars_inds(vector_indices(vars));
    for (size_t i(0); i < nrecords; ++i) {
      double normalizer(norm_2(vector_data[i](vars_inds)));
      if (normalizer == 0)
        continue;
      foreach(size_t j2, vars_inds)
        vector_data[i](j2) /= normalizer;
    }
  }

  void vector_dataset::randomize(double random_seed) {
    boost::mt11213b rng(static_cast<unsigned>(random_seed));
    std::vector<size_t> fin_tmp(finite_seq.size());
    vec vec_tmp;
    vec_tmp.resize(dvector);
    double weight_tmp;
    for (size_t i = 0; i < nrecords-1; ++i) {
      size_t j = (size_t)(boost::uniform_int<int>(i,nrecords-1)(rng));
      sill::copy(finite_data[i], fin_tmp.begin());
      sill::copy(finite_data[j], finite_data[i].begin());
      sill::copy(fin_tmp, finite_data[j].begin());
      sill::copy(vector_data[i], vec_tmp.begin());
      sill::copy(vector_data[j], vector_data[i].begin());
      sill::copy(vec_tmp, vector_data[j].begin());
      if (weighted) {
        weight_tmp = weights_[i];
        weights_[i] = weights_[j];
        weights_[j] = weight_tmp;
      }
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>
