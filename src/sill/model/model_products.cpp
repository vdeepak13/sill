
#include <sill/model/model_products.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  void model_product_inplace(const crf_model<table_crf_factor>& YgivenXmodel,
                             decomposable<table_factor>& YXmodel) {
    if (!set_equal(YgivenXmodel.input_arguments(), YXmodel.arguments())) {
      throw std::invalid_argument
        ("model_product_inplace() given models P(Y|X), P(X) with non-matching X argument sets; size(X) for P(Y|X) is " + to_string(YgivenXmodel.input_arguments().size()) + ", size for P(X) is " + to_string(YXmodel.arguments().size()) + "\nX from P(Y|X): " + to_string(YgivenXmodel.input_arguments()) + "\nX from P(X): " + to_string(YXmodel.arguments()) + "\n");
    }
    std::vector<table_factor> factors;
    foreach(const table_crf_factor& f, YgivenXmodel.factors()) {
      if (f.log_space()) {
        table_factor tmpf(f.get_table());
        tmpf.update(exponent<double>());
        factors.push_back(tmpf);
      } else {
        factors.push_back(f.get_table());
      }
    }
    YXmodel *= factors;
  }

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
