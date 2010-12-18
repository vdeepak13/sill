
#ifndef SILL_DISCRIMINATIVE_LOAD_FUNCTIONS_HPP
#define SILL_DISCRIMINATIVE_LOAD_FUNCTIONS_HPP

#include <sill/learning/discriminative/binary_classifier.hpp>
#include <sill/learning/discriminative/multiclass_classifier.hpp>

#include <sill/macros_def.hpp>

/**
 * \file load_functions.hpp Functions for loading learners from files;
 *                          this deals with the issue of learners being
 *                          templated.
 */

namespace sill {

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
   */
  boost::shared_ptr<binary_classifier>
  load_binary_classifier(std::ifstream& in, const datasource& ds);

  /**
   * Input a binary classifier from a human-readable file.
   * @param filename  file holding the saved classifier
   * @param ds        datasource used to get variables and variable orderings
   */
  boost::shared_ptr<binary_classifier>
  load_binary_classifier(const std::string& filename, const datasource& ds);

  /**
   * Input a multiclass classifier from a human-readable file.
   * @param in   input filestream for file holding the saved classifier
   * @param ds   datasource used to get variables and variable orderings
   */
  boost::shared_ptr<multiclass_classifier>
  load_multiclass_classifier(std::ifstream& in, const datasource& ds);

  /**
   * Input a multiclass classifier from a human-readable file.
   * @param filename  file holding the saved classifier
   * @param ds        datasource used to get variables and variable orderings
   */
  boost::shared_ptr<multiclass_classifier>
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
   * @return  pointer to empty classifier; or NULL pointer if unable to load
   */
  boost::shared_ptr<sill::binary_classifier>
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
   * @return  pointer to empty classifier; or NULL pointer if unable to load
   */
  boost::shared_ptr<sill::multiclass_classifier>
  empty_multiclass_classifier(std::string learner_name, sill::universe& u,
                              size_t booster_iterations);

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_DISCRIMINATIVE_LOAD_FUNCTIONS_HPP
