#include <sill/learning/validation/crossval_parameters.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  crossval_parameters::crossval_parameters()
    : dim(1), nfolds(10), minvals(dim), maxvals(dim),
      nvals(dim), zoom(0), log_scale(true),
      run_combo_type(statistics::MEAN) {
    minvals.fill(0.000001);
    maxvals.fill(1);
    nvals.fill(10);
  }

  crossval_parameters::crossval_parameters(size_t dim)
    : dim(dim), nfolds(10), minvals(dim), maxvals(dim),
      nvals(dim), zoom(0), log_scale(true),
      run_combo_type(statistics::MEAN) {
    if (dim == 0) {
      throw std::runtime_error
        ("crossval_parameters cannot be given dim = 0.");
    }
    minvals.fill(0.000001);
    maxvals.fill(1);
    nvals.fill(10);
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
      if (minvals[i] < 0)
        return false;
      if (maxvals[i] < minvals[i])
        return false;
      if (nvals[i] <= 0)
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

  void crossval_parameters::print(std::ostream& out) const {
    out << "dim : " << dim << "\n"
        << "nfolds: " << nfolds << "\n"
        << "minvals: " << minvals << "\n"
        << "nvals: " << nvals << "\n"
        << "zoom: " << zoom << "\n"
        << "log_scale: " << log_scale << "\n"
        << "run_combo_type: "
        << statistics::generalized_mean_string(run_combo_type) << "\n";
  }

  std::ostream&
  operator<<(std::ostream& out, const crossval_parameters& cv_params) {
    cv_params.print(out);
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>
