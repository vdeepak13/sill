#ifndef SILL_RESULT_SET_HPP
#define SILL_RESULT_SET_HPP

#include <sill/base/stl_util.hpp>
#include <sill/math/linear_algebra.hpp>
#include <sill/math/statistics.hpp>

namespace sill {

  /**
   * Class for collecting and aggregating results.
   *  - Collection: Results are collected as a map from strings (result names)
   *                to lists of values.
   *  - Aggregation: This class has methods for printing statistics about
   *                 the collected results.
   */
  class result_set {

  public:

    result_set();

    //! Add result to result set.
    void insert(const std::string& result_name, double val);

    /**
     * Get a result [mean, deviation] pair.
     *
     * @param run_combo_type  Type of statistic used to combine the result
     *                        values for each result type.
     *                         (default = MEAN)
     */
    std::pair<double,double>
    get(const std::string& result_name,
        statistics::generalized_mean_enum run_combo_type
        = statistics::MEAN) const;

    /**
     * Print aggregated results.
     *
     * Print modes:
     *  - 0: For each result type, print:
     *         "[result name] mean: [mean value]\n"
     *         "[result name] stderr: [stderr value]\n"
     *  - 1: For each result type, print:
     *         "[result name]: [list of values]\n"
     *
     * @param print_mode      Specifies what to print.
     *                         (default = 0)
     * @param run_combo_type  Type of statistic used to combine the result
     *                        values for each result type.
     *                         (default = MEAN)
     */
    void print(std::ostream& out,
               size_t print_mode = 0,
               statistics::generalized_mean_enum run_combo_type
               = statistics::MEAN) const;

    //! Clear results.
    void clear();

    // Private types and data
    //==========================================================================
  private:

    struct string_comparator {
      template <typename CharT>
      bool operator()(const std::basic_string<CharT>& a,
                      const std::basic_string<CharT>& b) const {
        return a.compare(b);
      }
    }; // struct string_comparator

    std::map<std::string, std::vector<double> > results_;

  }; // class result_set

} // namespace sill

#endif // #ifndef SILL_RESULT_SET_HPP
