#ifndef SILL_MODEL_PRODUCTS_HPP
#define SILL_MODEL_PRODUCTS_HPP

#include <sill/factor/gaussian_crf_factor.hpp>
#include <sill/factor/table_crf_factor.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/crf_model.hpp>
#include <sill/model/decomposable.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Multiply models P(X) and P(Y|X) together to create a decomposable model
   * Q(Y,X); note that Q(Y,X) != P(X) P(Y|X) in general (because of the
   * normalization constant for P(X)).
   * WARNING: Both P(X) and P(Y|X) must be tractable models, and their product
   *          must be tractable as well.
   *
   * @param  YgivenXmodel   Model for P(Y|X)
   * @param  YXmodel        This must store P(X) initially, and upon return,
   *                        it stores Q(Y,X).
   */
  void model_product_inplace(const crf_model<table_crf_factor>& YgivenXmodel,
                             decomposable<table_factor>& YXmodel) {
    if (!set_equal(YgivenXmodel.input_arguments(), YXmodel.arguments())) {
      throw std::invalid_argument
        ("model_product_inplace() given models P(Y|X), P(X) with non-matching X argument sets; size(X) for P(Y|X) is " + to_string(YgivenXmodel.input_arguments().size()) + ", size for P(X) is " + to_string(YXmodel.arguments().size()) + "\nX from P(Y|X): " + to_string(YgivenXmodel.input_arguments()) + "\nX from P(X): " + to_string(YXmodel.arguments()) + "\n");
    }
    std::vector<table_factor> factors;
    foreach(const table_crf_factor& f, YgivenXmodel.factors())
      factors.push_back(f.get_table());
    YXmodel *= factors;
  }

  /**
   * Multiply models P(X) and P(Y|X) together to create a decomposable model
   * Q(Y,X); note that Q(Y,X) != P(X) P(Y|X) in general (because of the
   * normalization constant for P(X)).
   * WARNING: Both P(X) and P(Y|X) must be tractable models, and their product
   *          must be tractable as well.
   *
   * @param  YgivenXmodel   Model for P(Y|X)
   * @param  YXmodel        This must store P(X) initially, and upon return,
   *                        it stores Q(Y,X).
   */
  void model_product_inplace(const crf_model<gaussian_crf_factor>& YgivenXmodel,
                             decomposable<canonical_gaussian>& YXmodel) {
    if (!set_equal(YgivenXmodel.input_arguments(), YXmodel.arguments())) {
      throw std::invalid_argument
        ("model_product_inplace() given models P(Y|X), P(X) with non-matching X argument sets; size(X) for P(Y|X) is " + to_string(YgivenXmodel.input_arguments().size()) + ", size for P(X) is " + to_string(YXmodel.arguments().size()) + "\nX from P(Y|X): " + to_string(YgivenXmodel.input_arguments()) + "\nX from P(X): " + to_string(YXmodel.arguments()) + "\n");
    }
    std::vector<canonical_gaussian> factors;
    foreach(const gaussian_crf_factor& f, YgivenXmodel.factors())
      factors.push_back(f.get_gaussian<canonical_gaussian>());
    YXmodel *= factors;
  }

}  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_MODEL_PRODUCTS_HPP
