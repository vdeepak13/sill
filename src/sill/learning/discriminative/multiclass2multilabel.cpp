#include <sill/learning/discriminative/multiclass2multilabel.hpp>

namespace sill {

  // Parameters: public methods
  //==========================================================================

  void multiclass2multilabel_parameters::load(std::ifstream& in,
                                              const datasource& ds) {
    base_learner = load_multiclass_classifier(in, ds);
    std::string line;
    getline(in, line);
    std::istringstream is(line);
    if (!(is >> random_seed))
      assert(false);
  }

} // namespace sill
