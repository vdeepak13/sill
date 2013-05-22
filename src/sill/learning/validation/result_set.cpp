
#include <sill/learning/validation/result_set.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  result_set::result_set() { }

  void result_set::insert(const std::string& result_name, double val) {
    std::map<std::string, std::vector<double> >::iterator
      it(results_.find(result_name));
    if (it == results_.end()) {
      results_[result_name] = std::vector<double>(1, val);
    } else {
      it->second.push_back(val);
    }
  }

  std::pair<double,double>
  result_set::
  get(const std::string& result_name,
      statistics::generalized_mean_enum run_combo_type) const {
    double avg =
      generalized_mean(safe_get(results_, result_name), run_combo_type);
    double dev =
      generalized_deviation(safe_get(results_, result_name), run_combo_type);
    return std::make_pair(avg, dev);
  }

  void
  result_set::print(std::ostream& out, size_t print_mode,
                    statistics::generalized_mean_enum run_combo_type) const {
    std::vector<std::string> sorted_result_names;
    {
      std::set<std::string> resnames(keys(results_));
      sorted_result_names.insert(sorted_result_names.end(),
                                 resnames.begin(), resnames.end());
      std::sort(sorted_result_names.begin(), sorted_result_names.end(),
                string_comparator());
    }
    if (print_mode == 0) {
      foreach(const std::string& resname, sorted_result_names) {
        out << resname << " "
            << statistics::generalized_mean_string(run_combo_type) << ": "
            << generalized_mean(safe_get(results_,resname), run_combo_type)
            << "\n"
            << resname << " "
            << statistics::generalized_deviation_string(run_combo_type) << ": "
            << generalized_deviation(safe_get(results_,resname), run_combo_type)
            << "\n";
      }
    } else if (print_mode == 1) {
      foreach(const std::string& resname, sorted_result_names)
        out << resname << ": " << safe_get(results_,resname) << "\n";
    } else {
      assert(false); // bad print_mode
    }
  } // print

  void result_set::clear() {
    results_.clear();
  }

} // namespace sill

#include <sill/macros_undef.hpp>
