#ifndef SILL_DISCRIMINATIVE_LOAD_FUNCTIONS_HPP
#define SILL_DISCRIMINATIVE_LOAD_FUNCTIONS_HPP

#include <sill/base/universe.hpp>
#include <sill/learning/discriminative/logistic_regression.hpp>
#include <sill/learning/discriminative/multiclass_classifier.hpp>

#include <sill/macros_def.hpp>

/**
 * \file load_functions.hpp Functions for loading learners from files;
 *                          this deals with the issue of learners being
 *                          templated.
 */

namespace sill {

  // Forward declarations
  /*
  template <typename Objective> class batch_booster_OC;
  template <typename Objective> class batch_booster;
  template <typename Objective> class decision_tree;
  template <typename Objective> class filtering_booster;
  class logistic_regression;
  template <typename Objective> class stump;
  template <typename Objective> class haar;
  */

  // Free functions
  //==========================================================================

  /**
   * Parse a learner's name.
   * @param  learner name in the form 'LEARNER_NAME<OBJECTIVE>; BASE<OBJ>; ...'
   *         where OBJECTIVE and the base learners are optional
   * Note: It's OK to have the one semicolon and spaces between meta and base
   *       learners, but other characters are not OK.
   * @return <LEARNER_NAME, OBJECTIVE, base_list>
   */
  boost::tuple<std::string, std::string, std::string>
  parse_learner_name(const std::string& fullname);

  /**
   * Checks a list of known learners and returns info about them.
   * @param  name  name of learner
   * @return <is_binary, is_multiclass, is_multilabel, is_online>
   */
  boost::tuple<bool, bool, bool, bool>
  check_learner_info(const std::string& name);

  /**
   * Input a binary classifier from a human-readable file.
   * @param in   input filestream for file holding the saved classifier
   * @param ds   datasource used to get variables and variable orderings
   * @tparam LA  Linear algebra type specifier
   */
  template <typename LA>
  boost::shared_ptr<binary_classifier<LA> >
  load_binary_classifier(std::ifstream& in, const datasource& ds);

  /**
   * Input a binary classifier from a human-readable file.
   * @param filename  file holding the saved classifier
   * @param ds        datasource used to get variables and variable orderings
   * @tparam LA  Linear algebra type specifier
   */
  template <typename LA>
  boost::shared_ptr<binary_classifier<LA> >
  load_binary_classifier(const std::string& filename, const datasource& ds);

  /**
   * Input a multiclass classifier from a human-readable file.
   * @param in   input filestream for file holding the saved classifier
   * @param ds   datasource used to get variables and variable orderings
   * @tparam LA  Linear algebra type specifier
   */
  template <typename LA>
  boost::shared_ptr<multiclass_classifier<LA> >
  load_multiclass_classifier(std::ifstream& in, const datasource& ds);

  /**
   * Input a multiclass classifier from a human-readable file.
   * @param filename  file holding the saved classifier
   * @param ds        datasource used to get variables and variable orderings
   * @tparam LA  Linear algebra type specifier
   */
  template <typename LA>
  boost::shared_ptr<multiclass_classifier<LA> >
  load_multiclass_classifier(const std::string& filename,
                             const datasource& ds);

  /**
   * Create an empty binary classifier which may be used to train new
   * binary classifiers of that type.
   *
   * Note: This function is really for using scripts to run learners.
   *       You should write your own code if you want to do fancy stuff (like
   *       running a meta-learner which uses another meta-learner as its
   *       base learner).
   *
   * @param learner_name        See parse_learner_name() for the format.
   * @param booster_iterations  Number of booster iterations to use for this
   *                            and any base learners.
   * @tparam LA  Linear algebra type specifier
   * @return  pointer to empty classifier; or NULL pointer if unable to load
   */
  template <typename LA>
  boost::shared_ptr<sill::binary_classifier<LA> >
  empty_binary_classifier(std::string learner_name,
                          size_t booster_iterations);

  /**
   * Create an empty multiclass classifier which may be used to train new
   * multiclass classifiers of that type.
   *
   * Note: This function is really for using scripts to run learners.
   *       You should write your own code if you want to do fancy stuff (like
   *       running a meta-learner which uses another meta-learner as its
   *       base learner).
   *
   * @tparam LA  Linear algebra type specifier
   * @return  pointer to empty classifier; or NULL pointer if unable to load
   */
  template <typename LA>
  boost::shared_ptr<sill::multiclass_classifier<LA> >
  empty_multiclass_classifier(std::string learner_name, sill::universe& u,
                              size_t booster_iterations);


  //============================================================================
  // Implementations of the above functions
  //============================================================================

  template <typename LA>
  boost::shared_ptr<binary_classifier<LA> >
  load_binary_classifier(std::ifstream& in, const datasource& ds) {
    std::string line;
    getline(in, line);
    boost::shared_ptr<binary_classifier<LA> > ptr;
    std::string name, obj, base;
    boost::tie(name, obj, base) = parse_learner_name(line);
    if (base.size() > 0) {
      std::cerr << "load_binary_classifier() was called with an invalid"
                << " learner name: \"" << line << "\"" << std::endl;
      assert(false);
      return ptr;
    }
    if (name.compare("logistic_regression") == 0) {
      ptr.reset(new logistic_regression<LA>());
    } else {
      std::cerr << "load_binary_classifier() did not recognize the classifier "
                << "name: " << name << std::endl;
      assert(false);
    }
    ptr->load(in, ds, false);
    return ptr;
  }

  template <typename LA>
  boost::shared_ptr<binary_classifier<LA> >
  load_binary_classifier(const std::string& filename, const datasource& ds) {
    std::ifstream in(filename.c_str(), std::ios::in);
    boost::shared_ptr<binary_classifier<LA> > c
      = load_binary_classifier<LA>(in, ds);
    in.close();
    return c;
  }

  template <typename LA>
  boost::shared_ptr<multiclass_classifier<LA> >
  load_multiclass_classifier(std::ifstream& in, const datasource& ds) {
    std::string line;
    getline(in, line);
    boost::shared_ptr<multiclass_classifier<LA> > ptr;
    std::string name, obj, base;
    boost::tie(name, obj, base) = parse_learner_name(line);
    if (base.size() > 0) {
      std::cerr << "load_multiclass_classifier() was called with an invalid"
                << " learner name: \"" << line << "\"" << std::endl;
      assert(false);
      return ptr;
    }
    assert(false);
    ptr->load(in, ds, false);
    return ptr;
  }

  template <typename LA>
  boost::shared_ptr<multiclass_classifier<LA> >
  load_multiclass_classifier(const std::string& filename,
                             const datasource& ds) {
    std::ifstream in(filename.c_str(), std::ios::in);
    boost::shared_ptr<multiclass_classifier<LA> > c
      = load_multiclass_classifier<LA>(in, ds);
    in.close();
    return c;
  }

  template <typename LA>
  boost::shared_ptr<sill::binary_classifier<LA> >
  empty_binary_classifier(std::string learner_name,
                          size_t booster_iterations) {
    std::string name, obj, base;
    boost::tie(name, obj, base) = parse_learner_name(learner_name);
    boost::shared_ptr<binary_classifier<LA> > learner_ptr;
    boost::shared_ptr<binary_classifier<LA> > base_learner_ptr;
    if (base.size() > 0) {
      base_learner_ptr = empty_binary_classifier<LA>(base, booster_iterations);
      if (base_learner_ptr.get() == NULL) {
        std::cerr << "empty_binary_classifier() called with bad learner name: "
                  << "\"" << learner_name << "\"" << std::endl;
        return learner_ptr;
      }
    }
    if (name.compare("logistic_regression") == 0) {
      learner_ptr.reset(new logistic_regression<LA>());
    } else if (name.size() > 0) {
      std::cerr << "Learner not supported yet: " << name << std::endl;
    } else
      assert(false);
    return learner_ptr;
  }

  template <typename LA>
  boost::shared_ptr<sill::multiclass_classifier<LA> >
  empty_multiclass_classifier(std::string learner_name, sill::universe& u,
                              size_t booster_iterations) {
    std::string name, obj, base;
    boost::tie(name, obj, base) = parse_learner_name(learner_name);
    boost::shared_ptr<multiclass_classifier<LA> > learner_ptr;
    bool base_learner_is_binary(false);
    boost::shared_ptr<binary_classifier<LA> > binary_base_ptr;
    boost::shared_ptr<multiclass_classifier<LA> > multiclass_base_ptr;
    if (base.size() > 0) {
      binary_base_ptr = empty_binary_classifier<LA>(base, booster_iterations);
      if (binary_base_ptr.use_count() > 0) {
        base_learner_is_binary = true;
      } else {
        multiclass_base_ptr
          = empty_multiclass_classifier<LA>(base, u, booster_iterations);
        if (multiclass_base_ptr.use_count() > 0) {
          base_learner_is_binary = false;
        } else {
          std::cerr << "Could not load base learner: " << base << std::endl;
          return learner_ptr;
        }
      }
    }
    assert(false);
    return learner_ptr;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_DISCRIMINATIVE_LOAD_FUNCTIONS_HPP
