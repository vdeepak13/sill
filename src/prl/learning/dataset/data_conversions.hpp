
#ifndef SILL_DATA_CONVERSIONS_HPP
#define SILL_DATA_CONVERSIONS_HPP

#include <iostream>

#include <boost/serialization/shared_ptr.hpp>

#include <sill/learning/dataset/concepts.hpp>
#include <sill/learning/dataset/oracle.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /*
  //! Returns the Euclidean distance between two records.
  //! Finite values which differ have distance 1.
  double euclidean_distance(const record& r1, const record& r2) {
    double d(0);
    const std::vector<size_t>& finite1 = r1.finite();
    const std::vector<size_t>& finite2 = r2.finite();
    const vec& vector1 = r1.vector();
    const vec& vector2 = r2.vector();
    if (finite1.size() != finite2.size() || vector1.size() != vector2.size()) {
      std::cerr << "euclidean_distance() was called with non-matching"
                << " (incomparable) records." << std::endl;
      assert(false);
      return - std::numeric_limits<double>::max();
    }
    for (size_t i = 0; i < finite1.size(); ++i)
      d += (finite1[i] == finite2[i] ? 0 : 1);
    for (size_t i = 0; i < vector1.size(); ++i)
      d += (vector1[i] - vector2[i])*(vector1[i] - vector2[i]);
    if (d < 0) { return 0; } // for numerical precision (necessary?)
    return sqrt(d);
  }
  */

  /**
   * Draw examples from the given oracle to build a dataset.
   */
  template <typename Dataset>
  static boost::shared_ptr<Dataset>
  oracle2dataset(oracle& o, size_t max_records) {
    concept_assert((sill::Dataset<Dataset>));
    boost::shared_ptr<Dataset> data_ptr
      (new Dataset(o.finite_list(), o.vector_list(),
                   o.variable_type_order()));
    data_ptr->set_finite_class_variables(o.finite_class_variables());
    data_ptr->set_vector_class_variables(o.vector_class_variables());
    for (size_t i = 0; i < max_records; i++) {
      if (o.next())
        data_ptr->insert(o.current().finite(), o.current().vector());
      else
        break;
    }
    return data_ptr;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_DATA_CONVERSIONS_HPP

