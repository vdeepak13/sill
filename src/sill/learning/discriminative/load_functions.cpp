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

  boost::shared_ptr<binary_classifier<> >
  load_binary_classifier(std::ifstream& in, const datasource& ds) {
    using namespace sill::discriminative;
    using namespace sill::boosting;
    std::string line;
    getline(in, line);
    boost::shared_ptr<binary_classifier<> > ptr;
    std::string name, obj, base;
    boost::tie(name, obj, base) = parse_learner_name(line);
    if (base.size() > 0) {
      std::cerr << "load_binary_classifier() was called with an invalid"
                << " learner name: \"" << line << "\"" << std::endl;
      assert(false);
      return ptr;
    }
    if (name.compare("stump") == 0) {
      if (obj.compare(objective_accuracy::name()) == 0)
        ptr.reset(new stump<objective_accuracy>());
      else if (obj.compare(objective_information::name()) == 0)
        ptr.reset(new stump<objective_information>());
      else
        assert(false);
    } else if (name.compare("decision_tree") == 0) {
      if (obj.compare(objective_accuracy::name()) == 0)
        ptr.reset(new decision_tree<objective_accuracy>());
      else if (obj.compare(objective_information::name()) == 0)
        ptr.reset(new decision_tree<objective_information>());
      else
        assert(false);
    } else if (name.compare("logistic_regression") == 0) {
      ptr.reset(new logistic_regression());
    } else if (name.compare("batch_booster") == 0) {
      if (obj.compare(adaboost::name()) == 0)
        ptr.reset(new batch_booster<adaboost>());
      else if (obj.compare(filterboost::name()) == 0)
        ptr.reset(new batch_booster<filterboost>());
      else
        assert(false);
    } else if (name.compare("filtering_booster")==0){
      if (obj.compare(adaboost::name()) == 0)
        ptr.reset(new filtering_booster<adaboost>());
      else if (obj.compare(filterboost::name()) == 0)
        ptr.reset(new filtering_booster<filterboost>());
      else
        assert(false);
    } else if (name.compare("haar") == 0) {
      if (obj.compare(objective_accuracy::name()) == 0)
        ptr.reset(new haar<objective_accuracy>());
      else if (obj.compare(objective_information::name()) == 0)
        ptr.reset(new haar<objective_information>());
      else
        assert(false);
    } else {
      std::cerr << "load_binary_classifier() did not recognize the classifier "
                << "name: " << name << std::endl;
      assert(false);
    }
    ptr->load(in, ds, false);
    return ptr;
  }

  boost::shared_ptr<binary_classifier<> >
  load_binary_classifier(const std::string& filename, const datasource& ds) {
    std::ifstream in(filename.c_str(), std::ios::in);
    boost::shared_ptr<binary_classifier<> > c
      = load_binary_classifier(in, ds);
    in.close();
    return c;
  }

  boost::shared_ptr<multiclass_classifier<> >
  load_multiclass_classifier(std::ifstream& in, const datasource& ds) {
    using namespace sill::discriminative;
    using namespace sill::boosting;
    std::string line;
    getline(in, line);
    boost::shared_ptr<multiclass_classifier<> > ptr;
    std::string name, obj, base;
    boost::tie(name, obj, base) = parse_learner_name(line);
    if (base.size() > 0) {
      std::cerr << "load_multiclass_classifier() was called with an invalid"
                << " learner name: \"" << line << "\"" << std::endl;
      assert(false);
      return ptr;
    }
    if (name.compare("batch_booster_OC") == 0) {
      if (obj.compare(adaboost::name()) == 0)
        ptr.reset(new batch_booster_OC<adaboost>());
      else if (obj.compare(filterboost::name()) == 0)
        ptr.reset(new batch_booster_OC<filterboost>());
      else
        assert(false);
    } else {
      assert(false);
    }
    ptr->load(in, ds, false);
    return ptr;
  }

  boost::shared_ptr<multiclass_classifier<> >
  load_multiclass_classifier(const std::string& filename,
                             const datasource& ds) {
    std::ifstream in(filename.c_str(), std::ios::in);
    boost::shared_ptr<multiclass_classifier<> > c
      = load_multiclass_classifier(in, ds);
    in.close();
    return c;
  }

  boost::shared_ptr<sill::binary_classifier<> >
  empty_binary_classifier(std::string learner_name,
                          size_t booster_iterations) {
    using namespace sill::discriminative;
    using namespace sill::boosting;
    std::string name, obj, base;
    boost::tie(name, obj, base) = parse_learner_name(learner_name);
    boost::shared_ptr<binary_classifier<> > learner_ptr;
    boost::shared_ptr<binary_classifier<> > base_learner_ptr;
    if (base.size() > 0) {
      base_learner_ptr = empty_binary_classifier(base, booster_iterations);
      if (base_learner_ptr.get() == NULL) {
        std::cerr << "empty_binary_classifier() called with bad learner name: "
                  << "\"" << learner_name << "\"" << std::endl;
        return learner_ptr;
      }
    }
    if (name.compare("stump") == 0) {
      if (obj.compare(objective_accuracy::name()) == 0)
        learner_ptr.reset(new stump<objective_accuracy>());
      else if (obj.compare(objective_information::name()) == 0)
        learner_ptr.reset(new stump<objective_information>());
      else if (obj.size() > 0)
        std::cerr << "Did not recognize objective: " << obj
                  << std::endl;
      else
        learner_ptr.reset(new stump<objective_accuracy>());
    } else if (name.compare("decision_tree") == 0) {
      if (obj.compare(objective_accuracy::name()) == 0)
        learner_ptr.reset(new decision_tree<objective_accuracy>());
      else if (obj.compare(objective_information::name()) == 0)
        learner_ptr.reset(new decision_tree<objective_information>());
      else if (obj.size() > 0)
        std::cerr << "Did not recognize objective: " << obj
                  << std::endl;
      else
        learner_ptr.reset
          (new decision_tree<objective_accuracy>());
    } else if (name.compare("logistic_regression") == 0) {
      learner_ptr.reset(new logistic_regression());
    } else if (name.compare("haar") == 0) {
      if (obj.compare(objective_accuracy::name()) == 0)
        learner_ptr.reset(new haar<objective_accuracy>());
      else if (obj.compare(objective_information::name()) == 0)
        learner_ptr.reset(new haar<objective_information>());
      else if (obj.size() > 0)
        std::cerr << "Did not recognize objective: " << obj
                  << std::endl;
      else
        learner_ptr.reset(new haar<objective_accuracy>());
    } else if (name.compare("batch_booster") == 0) {
      batch_booster_parameters params;
      params.init_iterations = booster_iterations;
      params.weak_learner = base_learner_ptr;
      if (obj.compare(adaboost::name()) == 0 || obj.size()==0){
        learner_ptr.reset(new batch_booster<adaboost>(params));
      } else if (obj.compare(filterboost::name()) == 0) {
        learner_ptr.reset(new batch_booster<filterboost>(params));
      } else {
        std::cerr << "Did not recognize booster objective: "
                  << obj << std::endl;
        return learner_ptr;
      }
    } else if (name.compare("filtering_booster") == 0) {
      filtering_booster_parameters params;
      params.init_iterations = booster_iterations;
      params.weak_learner = base_learner_ptr;
      if (obj.compare(adaboost::name()) == 0 || obj.size()==0){
        learner_ptr.reset(new filtering_booster<adaboost>(params));
      } else if (obj.compare(filterboost::name()) == 0) {
        learner_ptr.reset(new filtering_booster<filterboost>
                          (params));
      } else {
        std::cerr << "Did not recognize booster objective: "
                  << obj << std::endl;
        return learner_ptr;
      }
    } else if (name.size() > 0) {
      std::cerr << "Learner not supported yet: " << name << std::endl;
    } else
      assert(false);
    return learner_ptr;
  }

  boost::shared_ptr<sill::multiclass_classifier<> >
  empty_multiclass_classifier(std::string learner_name, sill::universe& u,
                              size_t booster_iterations) {
    using namespace sill::boosting;
    std::string name, obj, base;
    boost::tie(name, obj, base) = parse_learner_name(learner_name);
    boost::shared_ptr<multiclass_classifier<> > learner_ptr;
    bool base_learner_is_binary(false);
    boost::shared_ptr<binary_classifier<> > binary_base_ptr;
    boost::shared_ptr<multiclass_classifier<> > multiclass_base_ptr;
    if (base.size() > 0) {
      binary_base_ptr = empty_binary_classifier(base, booster_iterations);
      if (binary_base_ptr.use_count() > 0) {
        base_learner_is_binary = true;
      } else {
        multiclass_base_ptr
          = empty_multiclass_classifier(base, u, booster_iterations);
        if (multiclass_base_ptr.use_count() > 0) {
          base_learner_is_binary = false;
        } else {
          std::cerr << "Could not load base learner: " << base << std::endl;
          return learner_ptr;
        }
      }
    }
    if (learner_name.compare("batch_booster_OC") == 0) {
      batch_booster_OC_parameters params;
      if (!base_learner_is_binary) {
        assert(false);
        return learner_ptr;
      }
      params.init_iterations = booster_iterations;
      params.binary_label = u.new_finite_variable(2);
      params.weak_learner = binary_base_ptr;
      if (obj.compare(adaboost::name()) == 0 || obj.size() == 0) {
        learner_ptr.reset(new batch_booster_OC<adaboost>(params));
      } else if (obj.compare(filterboost::name()) == 0) {
        learner_ptr.reset(new batch_booster_OC<filterboost>(params));
      } else {
        std::cerr << "Did not recognize booster objective: "
                  << obj << std::endl;
        return learner_ptr;
      }
    } else if (learner_name.compare("all_pairs_batch") == 0) {
      all_pairs_batch_parameters params;
      if (!base_learner_is_binary) {
        assert(false);
        return learner_ptr;
      }
      params.binary_label = u.new_finite_variable(2);
      params.base_learner = binary_base_ptr;
      learner_ptr.reset(new all_pairs_batch(params));
    } else
      assert(false);
    return learner_ptr;
  }

} // namespace sill

#include <sill/macros_undef.hpp>
