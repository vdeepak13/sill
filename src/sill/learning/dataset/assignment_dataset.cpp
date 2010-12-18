#include <sill/learning/dataset/assignment_dataset.hpp>
#include <sill/math/norms.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Protected methods required by record
  //==========================================================================

  void assignment_dataset::load_assignment(size_t i, sill::assignment& a) const {
    assert(i < nrecords);
    a = data_vector[i];
  }

  void assignment_dataset::load_record(size_t i, record& r) const {
    if (!r.fin_own) {
      r.fin_own = true;
      r.fin_ptr = new std::vector<size_t>(finite_numbering_ptr_->size());
    }
    if (!r.vec_own) {
      r.vec_own = true;
      r.vec_ptr = new vec(dvector);
    }
    convert_finite_assignment2record(data_vector[i].finite(), *(r.fin_ptr));
    convert_vector_assignment2record(data_vector[i].vector(), *(r.vec_ptr));
  }

  void
  assignment_dataset::load_finite(size_t i,
                                  std::vector<size_t>& findata) const {
    convert_finite_assignment2record(data_vector[i].finite(), findata);
  }

  void assignment_dataset::load_vector(size_t i, vec& vecdata) const {
    convert_vector_assignment2record(data_vector[i].vector(), vecdata);
  }

  void assignment_dataset::load_assignment_pointer(size_t i, assignment** a) const {
    assert(i < nrecords);
    *a = &(data_vector[i]);
  }

  // Protected functions
  //==========================================================================

  void assignment_dataset::init(size_t nreserved) {
    assert(nreserved > 0);
    data_vector.resize(nreserved);
    for (size_t j(0); j < vector_seq.size(); ++j)
      for (size_t k(0); k < vector_seq[j]->size(); ++k)
        vector_i2pair.push_back(std::make_pair(vector_seq[j], k));
  }

  // Mutating operations
  //==========================================================================

  void assignment_dataset::reserve(size_t n) {
    if (n > capacity()) {
      data_vector.resize(n);
      if (weighted)
        weights_.resize(n, true);
    }
  }

  void assignment_dataset::set_record(size_t i, const assignment& a, double w) {
    assert(i < nrecords);
    data_vector[i].clear();
    foreach(finite_variable* v, finite_seq) {
      finite_assignment::const_iterator it(a.finite().find(v));
      assert(it != a.finite().end());
      data_vector[i].finite()[v] = it->second;
    }
    foreach(vector_variable* v, vector_seq) {
      vector_assignment::const_iterator it(a.vector().find(v));
      assert(it != a.vector().end());
      data_vector[i].vector()[v] = it->second;
    }
//    data_vector[i] = a;
    if (weighted) {
      weights_[i] = w;
    } else if (w != 1) {
      std::cerr << "assignment_dataset::set_record() called with weight"
                << " w != 1 on an unweighted dataset." << std::endl;
      assert(false);
    }
  }

  void
  assignment_dataset::set_record(size_t i, const std::vector<size_t>& fvals,
                                 const vec& vvals, double w) {
    assert(i < nrecords);
    convert_finite_record2assignment(fvals, data_vector[i].finite());
    convert_vector_record2assignment(vvals, data_vector[i].vector());
    if (weighted) {
      weights_[i] = w;
    } else if (w != 1) {
      std::cerr << "assignment_dataset::set_record() called with weight"
                << " w != 1 on an unweighted dataset." << std::endl;
      assert(false);
    }
  }

  void assignment_dataset::normalize(const vec& means,
                                     const vec& std_devs) {
    assert(means.size() == dvector);
    assert(std_devs.size() == dvector);
    for (size_t j(0); j < dvector; ++j) {
      const std::pair<vector_variable*, size_t>& ipair = vector_i2pair[j];
      if (std_devs[j] < 0)
        assert(false);
      double std_dev(std_devs[j] == 0 ? 1 : std_devs[j]);
      for (size_t i(0); i < nrecords; ++i)
        data_vector[i].vector()[ipair.first][ipair.second] =
          (data_vector[i].vector()[ipair.first][ipair.second] - means[j])
          / std_dev;
    }
  }

  void assignment_dataset::normalize(const vec& means, const vec& std_devs,
                                     const vector_var_vector& vars) {
    foreach(vector_variable* v, vars)
      assert(vector_vars.count(v) != 0);
    ivec vars_inds(vector_indices(vars));
    assert(means.size() == vars_inds.size());
    assert(std_devs.size() == vars_inds.size());
    for (size_t j(0); j < vars_inds.size(); ++j) {
      size_t j2(static_cast<size_t>(vars_inds[j]));
      const std::pair<vector_variable*, size_t>& ipair = vector_i2pair[j2];
      if (std_devs[j] < 0)
        assert(false);
      double std_dev(std_devs[j] == 0 ? 1 : std_devs[j]);
      for (size_t i(0); i < nrecords; ++i)
        data_vector[i].vector()[ipair.first][ipair.second] =
          (data_vector[i].vector()[ipair.first][ipair.second] - means[j])
          / std_dev;
    }
  }

  void assignment_dataset::normalize2(const vector_var_vector& vars) {
    foreach(vector_variable* v, vars)
      assert(vector_vars.count(v) != 0);
    ivec vars_inds(vector_indices(vars));
    for (size_t i(0); i < nrecords; ++i) {
      double normalizer(0);
      foreach(vector_variable* v, vars) {
        const vec& tmpval = data_vector[i].vector()[v];
        normalizer += tmpval.inner_prod(tmpval);
      }
      if (normalizer == 0)
        continue;
      normalizer = sqrt(normalizer);
      foreach(vector_variable* v, vars)
        data_vector[i].vector()[v] /= normalizer;
    }
  }

  void assignment_dataset::randomize(double random_seed) {
    boost::mt11213b rng(static_cast<unsigned>(random_seed));
    assignment tmpa;
    double weight_tmp;
    for (size_t i = 0; i < nrecords-1; ++i) {
      size_t j = (size_t)(boost::uniform_int<int>(i,nrecords-1)(rng));
      tmpa = data_vector[i];
      data_vector[i] = data_vector[j];
      data_vector[j] = tmpa;
      if (weighted) {
        weight_tmp = weights_[i];
        weights_[i] = weights_[j];
        weights_[j] = weight_tmp;
      }
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>
