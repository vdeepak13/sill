#include <sill/learning/discriminative/all_pairs_batch.hpp>
#include <sill/learning/discriminative/batch_booster.hpp>
#include <sill/learning/discriminative/batch_booster_OC.hpp>
#include <sill/learning/discriminative/boosters.hpp>
//#include <sill/learning/discriminative/concepts.hpp>
#include <sill/learning/discriminative/decision_tree.hpp>
#include <sill/learning/discriminative/filtering_booster.hpp>
#include <sill/learning/discriminative/load_functions.hpp>
#include <sill/learning/discriminative/logistic_regression.hpp>
#include <sill/learning/discriminative/stump.hpp>
#include <sill/learning/object_detection/haar.hpp>
#include <sill/base/universe.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Free functions
  //==========================================================================

  boost::tuple<std::string, std::string, std::string>
  parse_learner_name(const std::string& fullname) {
    size_t left(fullname.find_first_of("<"));
    size_t right(fullname.find_first_of(">"));
    size_t semi(fullname.find_first_of(";"));
    std::string name(fullname.substr(0, left));
    std::string obj, base;
    if (left == std::string::npos) {
      if (right != std::string::npos) {
        std::cerr << "parse_learner_name() could not parse learner name: \""
                  << fullname << "\"" << std::endl;
        assert(false);
        return boost::tuple<std::string, std::string, std::string>();
      }
    } else {
      if (right == std::string::npos) {
        std::cerr << "parse_learner_name() could not parse learner name: \""
                  << fullname << "\"" << std::endl;
        assert(false);
        return boost::tuple<std::string, std::string, std::string>();
      }
      obj = fullname.substr(left+1, right - left - 1);
    }
    if (semi != std::string::npos) {
      if (semi < right) {
        std::cerr << "parse_learner_name() could not parse learner name: \""
                  << fullname << "\"" << std::endl;
        assert(false);
        return boost::tuple<std::string, std::string, std::string>();
      }
      ++semi;
      while (fullname[semi] == ' ')
        ++semi;
      if (semi >= fullname.size()) {
        std::cerr << "parse_learner_name() could not parse learner name: \""
                  << fullname << "\"\n"
                  << "(Did you have a hanging semicolon?)"<< std::endl;
        assert(false);
        return boost::tuple<std::string, std::string, std::string>();
      }
      base = fullname.substr(semi, fullname.size() - semi);
    }
    return boost::make_tuple(name, obj, base);
  }

  boost::tuple<bool, bool, bool, bool>
  check_learner_info(const std::string& name) {
    if (name.compare("all_pairs_batch") == 0) {
      return boost::make_tuple(false, true, false, false);
    } else if (name.compare("batch_booster") == 0) {
      return boost::make_tuple(true, false, false, false);
    } else if (name.compare("batch_booster_OC") == 0) {
      return boost::make_tuple(false, true, false, false);
    } else if (name.compare("decision_tree") == 0) {
      return boost::make_tuple(true, false, false, false);
    } else if (name.compare("filtering_booster") == 0) {
      return boost::make_tuple(true, false, false, true);
    } else if (name.compare("filtering_booster_OC") == 0) {
      return boost::make_tuple(false, true, false, true);
    } else if (name.compare("logistic_regression") == 0) {
      // TODO: THIS ISN'T QUITE CORRECT
      return boost::make_tuple(true, false, false, false);
    } else if (name.compare("stump") == 0) {
      return boost::make_tuple(true, false, false, false);
    } else if (name.compare("haar") == 0) {
      return boost::make_tuple(true, false, false, false);
    } else {
      std::cerr << "check_learner_info() did not recognize learner name: \""
                << name << "\"" << std::endl;
      assert(false);
      return boost::make_tuple(false, false, false, false);
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>
