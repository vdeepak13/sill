
#include <sill/learning/validation/crossval_parameters.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  crossval_parameters::crossval_parameters()
    : dim(1), nfolds(10), minvals(dim, 0.000001), maxvals(dim, 1),
      nvals(dim, 10), zoom(0), log_scale(true),
      run_combo_type(statistics::MEAN) { }

  crossval_parameters::crossval_parameters(size_t dim)
    : dim(dim), nfolds(10), minvals(dim, 0.000001), maxvals(dim, 1),
      nvals(dim, 10), zoom(0), log_scale(true),
      run_combo_type(statistics::MEAN) {
    if (dim == 0) {
      throw std::runtime_error
        ("crossval_parameters cannot be given dim = 0.");
    }
  }

  bool crossval_parameters::valid() const {
    if (dim == 0)
      return false;
    if (nfolds <= 1)
      return false;
    if (minvals.size() != dim)
      return false;
    if (maxvals.size() != dim)
      return false;
    if (nvals.size() != dim)
      return false;
    for (size_t i(0); i < dim; ++i) {
      if (minvals(i) < 0)
        return false;
      if (maxvals(i) < minvals(i))
        return false;
      if (nvals(i) <= 0)
        return false;
    }
    return true;
  }

  void crossval_parameters::save(oarchive& ar) const {
    ar << dim << nfolds << minvals << maxvals << nvals << zoom << log_scale
       << run_combo_type;
  }

  void crossval_parameters::load(iarchive& ar) {
    ar >> dim >> nfolds >> minvals >> maxvals >> nvals >> zoom >> log_scale
       >> run_combo_type;
  }

} // namespace sill

#include <sill/macros_undef.hpp>
