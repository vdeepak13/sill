
#ifndef PRL_OBJECT_DETECTION_FREE_FUNCTIONS_HPP
#define PRL_OBJECT_DETECTION_FREE_FUNCTIONS_HPP

#include <prl/assignment.hpp>
#include <prl/learning/dataset/dataset.hpp>
#include <prl/learning/dataset/statistics.hpp>
#include <prl/learning/discriminative/concepts.hpp>
#include <prl/learning/discriminative/binary_classifier.hpp>
#include <prl/learning/discriminative/discriminative.hpp>
#include <prl/learning/object_detection/image.hpp>
#include <prl/stl_io.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * Convert a dataset which has records in a non-image format into a
   * dataset with records in image format.
   * The given dataset's variables must either be vector variables corresponding
   * to pixels or be finite variables.  The order in which the pixels are given
   * is determined by 'by_row.'
   * This does not copy dataset weights.
   *
   * @param u       universe
   * @param ds      original (non-image format) dataset
   * @param by_row  true = pixels are ordered left-to-right, then top-to-bottom;
   *                false = top-to-bottom, then left-to-right
   * @param height  height of images
   * @param width   width of images
   * @param depth   depth of images (default = 1 or grayscale)
   * @return dataset with image records
   */
  template <typename Dataset>
  boost::shared_ptr<Dataset>
  ds2image_format(universe& u, const Dataset& ds, bool by_row, size_t height,
                  size_t width, size_t depth) {
    concept_assert((prl::MutableDataset<Dataset>));
    assert(width > 0);
    assert(ds.vector_class_variables().size() == 0);
    vector_var_vector vector_vars
      (image::create_var_order(u, height, width, depth));
    const finite_var_vector& finite_vars = ds.finite_list();
    std::vector<variable::variable_typenames> var_type_order(vector_vars.size(),
                                                   variable::VECTOR_VARIABLE);
    for (size_t j = 0; j < finite_vars; ++j)
      var_type_order.push_back(variable::FINITE_VARIABLE);
    boost::shared_ptr<Dataset> ds_ptr(new Dataset(finite_vars, vector_vars,
                                                  var_type_order, ds.size()));
    vec vvals(ds_ptr->vector_dim());
    const vector_var_vector& orig_vector_vars = ds.vector_list();
    for (size_t i = 0; i < ds.size(); ++i) {
      for (
      ds_ptr->insert(ds[i].finite(), );
    }
    return ds_ptr;
  }

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_OBJECT_DETECTION_FREE_FUNCTIONS_HPP
