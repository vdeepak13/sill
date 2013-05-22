#ifndef SILL_MODEL_PRODUCTS_HPP
#define SILL_MODEL_PRODUCTS_HPP

#include <sill/factor/gaussian_crf_factor.hpp>
#include <sill/factor/table_crf_factor.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/crf_model.hpp>
#include <sill/model/decomposable.hpp>

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
                             decomposable<table_factor>& YXmodel);

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
                             decomposable<canonical_gaussian>& YXmodel);

}  // namespace sill

#endif // SILL_MODEL_PRODUCTS_HPP
