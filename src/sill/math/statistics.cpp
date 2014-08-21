#include <sill/math/statistics.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Means, standard errors, medians, and MADs
  //============================================================================

  namespace statistics {

    std::string generalized_mean_string(generalized_mean_enum gm) {
      switch (gm) {
      case MEAN:
        return "mean";
      case MEDIAN:
        return "median";
      default:
        assert(false); return "";
      }
    }

    std::string generalized_deviation_string(generalized_mean_enum gm) {
      switch (gm) {
      case MEAN:
        return "stderr";
      case MEDIAN:
        return "MAD";
      default:
        assert(false); return "";
      }
    }

  } // namespace statistics

  oarchive& operator<<(oarchive& a, statistics::generalized_mean_enum gm) {
    a << (size_t)(gm);
    return a;
  }

  iarchive& operator>>(iarchive& a, statistics::generalized_mean_enum& gm) {
    size_t tmp;
    a >> tmp;
    gm = (statistics::generalized_mean_enum)(tmp);
    return a;
  }

  // Order statistics
  //============================================================================

  std::vector<size_t> sorted_indices(const std::vector<vec>& v) {
    std::vector<size_t> ind(v.size());
    for (size_t i(0); i < v.size(); ++i)
      ind[i] = i;
    impl::sorted_indices_comparator2 comp(v);
    std::sort(ind.begin(), ind.end(), comp);
    return ind;
  }

} // namespace sill

#include <sill/macros_undef.hpp>
