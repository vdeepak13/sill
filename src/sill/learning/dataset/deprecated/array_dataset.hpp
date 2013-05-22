
#ifndef SILL_ARRAY_DATASET_HPP
#define SILL_ARRAY_DATASET_HPP

#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include <boost/iterator.hpp>
#include <boost/multi_array.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <sill/assignment.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/learning/dataset/dataset.hpp>
#include <sill/range/algorithm.hpp>
#include <sill/range/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that represents a dataset stored in an array.
   * \warning This has been deprecated.  Use vector_dataset instead.
   *
   * \author Joseph Bradley, Stanislav Funiak
   * \ingroup learning_dataset
   * \todo serialization
   */
  class array_dataset : public dataset {

  public:
    //! Base class
    typedef dataset base;

  protected:
    //! The type for storing data points' finite variable values.
    //! (# data points x # finite variables)
    typedef boost::multi_array<size_t, 2> finite_array;

    //! Type for storing data points' vector variable values.
    //! (# data points x total dimensionality of vector variables
    typedef boost::multi_array<double, 2> vector_array;

    //! Table of finite values.
    finite_array finite_data;

    //! Table of vector values.
    vector_array vector_data;

    ////////////////// PROTECTED FUNCTIONS REQUIRED BY record /////////////////

    //! Load datapoint i into assignment a
    void load_assignment(size_t i, sill::assignment& a) const {
      assert(i < nrecords);
      finite_assignment& fa = a.finite();
      fa.clear();
      foreach(const finite_var_index_pair& p, finite_numbering_)
        fa[p.first] = finite_data[i][p.second];
      vector_assignment& va = a.vector();
      va.clear();
      foreach(const vector_var_index_pair& p, vector_numbering_) {
        vec tmpvec(p.first->size());
        for(size_t j = 0; j < p.first->size(); j++)
          tmpvec[j] = vector_data[i][j+p.second];
        va[p.first] = tmpvec;
      }
    }
    //! Load record i into r
    void load_record(size_t i, record& r) const {
      sill::copy(finite_data[i], r.fin_ptr->begin());
      sill::copy(vector_data[i], r.vec_ptr->begin());
    }
    //! Load finite data for datapoint i into findata
    void load_finite(size_t i, std::vector<size_t>& findata) const {
      sill::copy(finite_data[i], findata.begin());
    }
    //! Load vector data for datapoint i into vecdata
    void load_vector(size_t i, vec& vecdata) const {
      sill::copy(vector_data[i], vecdata.begin());
    }

    //////////////////////////// PUBLIC METHODS ///////////////////////////

  public:
    //! Constructor for empty dataset.
    array_dataset() { }

    //! Constructs the dataset with the given variables
    //! Note this does not let the dataset know the ordering of the variables
    //! in the data file, so var_order() cannot be used to load another
    //! dataset over the same variables.
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! \param nreserved the number of reserved entries
    array_dataset(const finite_domain& finite_vars,
                  const vector_domain& vector_vars,
                  size_t nreserved = 1)
      : base(finite_vars, vector_vars) {
      assert(nreserved > 0);
      finite_data.resize(boost::extents[nreserved][num_finite()]);
      vector_data.resize(boost::extents[nreserved][dvector]);
    }

    //! Constructs the datset with the given sequence of variables
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! @param var_type_order  Order of variable types in datasource's
    //!                        natural order
    array_dataset(const finite_var_vector& finite_vars,
                  const vector_var_vector& vector_vars,
                  const std::vector<variable_type_enum>& var_type_order,
                  size_t nreserved = 1)
      : base(finite_vars, vector_vars, var_type_order) {
      assert(nreserved > 0);
      finite_data.resize(boost::extents[nreserved][num_finite()]);
      vector_data.resize(boost::extents[nreserved][dvector]);
    }

    //! Return capacity
    size_t capacity() const { return finite_data.shape()[0]; }

    //! Element access: record i, finite variable j (in the order finite_list())
    //! NOTE: This is being used for tests but should not normally be used in
    //! practice.
    size_t finite(size_t i, size_t j) const {
      assert(i < nrecords && j < num_finite());
      return finite_data[i][j];
    }
    //! Element access: record i, vector variable j (in the order finite_list(),
    //! but with n-value vector variables using n indices j)
    //! NOTE: This is being used for tests but should not normally be used in
    //! practice.
    double vector(size_t i, size_t j) const {
      assert(i < nrecords && j < dvector);
      return vector_data[i][j];
    }

    ///////////////// Mutating operations ////////////////////

    //! Increases the capacity in anticipation of adding new elements.
    void reserve(size_t n) {
      if (n > capacity()) {
        finite_data.resize(boost::extents[n][finite_data.shape()[1]]);
        vector_data.resize(boost::extents[n][vector_data.shape()[1]]);
      }
    }

    //! Adds a new record with weight w (default 1)
    void insert(const assignment& a, double w = 1) {
      if (nrecords == capacity()) reserve(2*nrecords);
      const finite_assignment& fa = a.finite();
      assert(finite_vars.subset_of(fa.keys()));
      for (size_t i = 0; i < num_finite(); i++)
        finite_data[nrecords][i] = fa[finite_seq[i]];
      const vector_assignment& va = a.vector();
      assert(vector_vars.subset_of(va.keys()));
      for (size_t i = 0; i < num_vector(); i++) {
        size_t offset = vector_numbering_[vector_seq[i]];
        vec tmpvec(va[vector_seq[i]]);
        for (size_t j = 0; j < vector_seq[i]->size(); j++)
          vector_data[nrecords][offset + j] = tmpvec[j];
      }
      if (weighted) {
        if (nrecords >= weights_.size())
          weights_.resize(2*nrecords, true);
        weights_[nrecords] = w;
      } else if (w != 1) {
        weights_.resize(2*nrecords);
        for (size_t i = 0; i < nrecords; ++i)
          weights_[i] = 1;
        weights_[nrecords] = w;
      }
      nrecords++;
    }
    //! Adds a new record with weight w (default 1)
    void insert(const std::vector<size_t>& fvals, const vec& vvals,
                double w = 1) {
      if (finite_data.size() == nrecords) reserve(2*nrecords);
      sill::copy(fvals, finite_data[nrecords].begin());
      sill::copy(vvals, vector_data[nrecords].begin());
      if (weighted) {
        if (nrecords >= weights_.size())
          weights_.resize(2*nrecords, true);
        weights_[nrecords] = w;
      } else if (w != 1) {
        weights_.resize(2*nrecords);
        for (size_t i = 0; i < nrecords; ++i)
          weights_[i] = 1;
        weights_[nrecords] = w;
      }
      nrecords++;
    }

    //! Sets record with index i to this value and weight.
    void set_record(size_t i, const assignment& a, double w = 1) {
      assert(false);
    }
    //! Sets record with index i to this value and weight.
    void set_record(size_t i, const std::vector<size_t>& fvals,
                    const vec& vvals, double w = 1) {
      assert(false);
    }

    //! Normalizes the vector data so that the empirical mean and variance
    //! are 0 and 1, respectively.
    //! \todo Should this take record weights into account?
    //! @return pair<means, std_devs>
    std::pair<std::vector<double>,std::vector<double> >
    normalize() {
      std::vector<double> means(dvector,0);
      std::vector<double> std_devs(dvector,0);
      for (size_t i = 0; i < nrecords; ++i)
        for (size_t j = 0; j < dvector; ++j) {
          means[j] += vector_data[i][j];
          std_devs[j] += vector_data[i][j] * vector_data[i][j];
        }
      for (size_t j = 0; j < dvector; ++j) {
        means[j] /= nrecords;
        std_devs[j] /= nrecords;
        std_devs[j] = sqrt(std_devs[j] - means[j] * means[j]);
        for (size_t i = 0; i < nrecords; ++i)
          vector_data[i][j] -= means[j];
        assert(std_devs[j] >= 0);
        if (std_devs[j] != 0)
          for (size_t i = 0; i < nrecords; ++i)
            vector_data[i][j] /= std_devs[j];
      }
      return std::make_pair(means, std_devs);
    }
    //! Normalizes the vector data using the given means and std_devs.
    //! \todo Should this take record weights into account?
    void normalize(std::vector<double> means,
                   std::vector<double> std_devs) {
      for (size_t j = 0; j < dvector; ++j) {
        for (size_t i = 0; i < nrecords; ++i)
          vector_data[i][j] -= means[j];
        assert(std_devs[j] >= 0);
        if (std_devs[j] != 0)
          for (size_t i = 0; i < nrecords; ++i)
            vector_data[i][j] /= std_devs[j];
      }
    }

    //! Clears the dataset of all records.
    //! NOTE: This should not be called if views of the data exist!
    //!       (This is unsafe but very useful for avoiding reallocation.)
    void clear() {
      nrecords = 0;
    }

    //! Randomly reorders the dataset (this is a mutable operation)
    void randomize(double random_seed) {
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

  };  // class array_dataset

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
