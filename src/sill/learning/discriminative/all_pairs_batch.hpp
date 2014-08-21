#ifndef SILL_LEARNING_DISCRIMINATIVE_ALL_PAIRS_BATCH_HPP
#define SILL_LEARNING_DISCRIMINATIVE_ALL_PAIRS_BATCH_HPP

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
//#include <boost/random/bernoulli_distribution.hpp>

//#include <sill/datastructure/concepts.hpp>
#include <sill/learning/discriminative/binary_classifier.hpp>
//#include <sill/learning/discriminative/concepts.hpp>
#include <sill/learning/discriminative/multiclass_classifier.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
//#include <sill/learning/dataset/oracle.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/math/statistics.hpp>

#include <sill/macros_def.hpp>

// Set to true to print debugging information.
#define DEBUG_ALL_PAIRS_BATCH 0

namespace sill {

  // Forward declarations
  /*
  template <typename LA>
  boost::shared_ptr<binary_classifier<LA> >
  load_binary_classifier(std::ifstream& in, const datasource& ds);
  */

  /**
   * @todo Add parameter NO_TRAIN_ACCURACY (for faster training).
   */
  struct all_pairs_batch_parameters {

    typedef dense_linear_algebra<> la_type;

    //! Specifies base learner type.
    //!  (required)
    boost::shared_ptr<binary_classifier<la_type> > base_learner;

    //! New variable used to create binary views of the class variable
    //!  (required)
    finite_variable* binary_label;

    //! Used to make the algorithm, including the base classifier, deterministic
    //!  (default = random)
    double random_seed;

    all_pairs_batch_parameters() : binary_label(NULL) {
      std::time_t time_tmp;
      time(&time_tmp);
      random_seed = time_tmp;
    }

    bool valid() const {
      if (binary_label == NULL || binary_label->size() == 2)
        return false;
      return true;
    }

    void save(std::ofstream& out) const {
      if (base_learner.get() == NULL) {
        std::cerr << "Error: cannot save all_pairs_batch which does not have a"
                  << " base learner type." << std::endl;
        assert(false);
        return;
      }
      base_learner->save(out);
      out << random_seed << "\n";
    }

    void load(std::ifstream& in, const datasource& ds) {
      assert(false); // TO BE FIXED
//      base_learner = load_binary_classifier<la_type>(in, ds);
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> random_seed))
        assert(false);
    }

  }; // struct all_pairs_batch_parameters

  /**
   * Creates a batch multiclass classifier from a batch binary classifier using
   * the all-pairs approach:
   *  - For all pairs of labels, a binary classifier is trained to distinguish
   *    between the pair (on only the data records which have one of the two
   *    labels).
   *  - To classify a new example, all pairs of possible labels are compared,
   *    and the label which wins out most often is chosen (with ties broken
   *    uniformly at random).
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @see BatchMulticlassClassifier
   * @todo Make a copy constructor and assignment operator which copy WLs.
   */
  class all_pairs_batch
    : public multiclass_classifier<dense_linear_algebra<> > {

    // Public types
    //==========================================================================
  public:

    typedef dense_linear_algebra<> la_type;

    typedef multiclass_classifier<la_type> base;

    typedef base::record_type record_type;

    typedef all_pairs_batch_parameters parameters;

    // Constructors
    //==========================================================================
  public:

    /**
     * Constructor without associated data; useful for:
     *  - creating other instances
     *  - loading a saved booster
     * @param params        algorithm parameters
     */
    explicit all_pairs_batch(parameters params = parameters())
      : params(params) {
    }

    /**
     * Constructor for an all-pairs batch classifier.
     * @param stats         a statistics class for the training dataset
     * @param parameters    algorithm parameters
     */
    explicit all_pairs_batch(dataset_statistics<la_type>& stats,
                             parameters params = parameters())
      : base(stats.get_dataset()), params(params) {
      build(stats);
    }

    /**
     * Constructor for an all-pairs batch classifier.
     * @param o    data oracle
     * @param n    max number of examples which should be drawn from the oracle
     * @param parameters    algorithm parameters
     */
    all_pairs_batch(oracle<la_type>& o, size_t n,
                    parameters params = parameters())
    : base(o), params(params) {
      boost::shared_ptr<vector_dataset<la_type> >
        ds_ptr(new vector_dataset<la_type>());
      oracle2dataset(o, n, *ds_ptr);
      dataset_statistics<la_type> stats(*ds_ptr);
      build(stats);
    }

    //! Train a new multiclass classifier of this type with the given data.
    boost::shared_ptr<multiclass_classifier<la_type> >
    create(dataset_statistics<la_type>& stats) const {
      boost::shared_ptr<multiclass_classifier<la_type> >
        bptr(new all_pairs_batch(stats, this->params));
      return bptr;
    }

    //! Train a new multiclass classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    boost::shared_ptr<multiclass_classifier<la_type> >
    create(oracle<la_type>& o, size_t n) const {
      boost::shared_ptr<multiclass_classifier<la_type> >
        bptr(new all_pairs_batch(o, n, this->params));
      return bptr;
    }

    // Getters and helpers
    //==========================================================================

    using base::nclasses;

    //! Return a name for the algorithm without template parameters.
    std::string name() const {
      return "all_pairs_batch";
    }

    //! Return a name for the algorithm with comma-separated template parameters
    //! (e.g., objective).
    std::string fullname() const {
      return name();
    }

    //! Returns true iff learner is naturally an online learner.
    bool is_online() const {
      return false;
    }

    //! Get the parameters of the algorithm
    parameters& get_parameters() {
      return params;
    }

    // Learning and mutating operations
    //==========================================================================

    //! Reset the random seed in this algorithm's parameters and in its
    //! random number generator.
    void random_seed(double value) {
      params.random_seed = value;
      rng.seed(static_cast<unsigned>(params.random_seed));
    }

    // Prediction methods
    //==========================================================================

    //! Predict the label of a new example.
    std::size_t predict(const record_type& example) const;

    //! Predict the label of a new example.
    std::size_t predict(const assignment& example) const;

    // Save and load methods
    //==========================================================================

    using base::save;
    using base::load;

    //! Output the classifier to a human-readable file which can be reloaded.
    //! @param save_part  0: save function (default), 1: engine, 2: shell
    //! @param save_name  If true, this saves the name of the learner.
    void save(std::ofstream& out, size_t save_part = 0,
              bool save_name = true) const;

    /**
     * Input the classifier from a human-readable file.
     * @param in          input filestream for file holding the saved learner
     * @param ds          datasource used to get variables
     * @param load_part   0: load function (default), 1: engine, 2: shell
     * This assumes that the learner name has already been checked.
     * @return true if successful
     */
    bool load(std::ifstream& in, const datasource& ds, size_t load_part);

    // Protected data members
    //==========================================================================
  protected:

    // Data from base class:
    using base::label_;
    using base::label_index_;

    parameters params;

    //! random number generator
    mutable boost::mt11213b rng;

    //! uniform distribution over [0, 1]
    mutable boost::uniform_real<double> uniform_prob;

    //! Arity of class variable
    size_t nclasses_;

    /**
     * base_classifiers[i][j] = classifier for comparing labels
     *  i (0), j+i+1 (1)
     * Note: i ranges from 0 to nclasses_-2;
     *       j ranges from 0 to nclasses_-i-2
     */
    std::vector<std::vector<boost::shared_ptr<binary_classifier<la_type> > > >
      base_classifiers;

    //! Training accuracies of base classifiers
    std::vector<std::vector<double> > base_train_acc;

    // Protected methods
    //==========================================================================

    void init(const datasource& ds);

    //! Train learner.
    void build(dataset_statistics<la_type>& stats);

    /*
    //! Train learner.
    void build_online(oracle<la_type>& o);
    */

  };  // all_pairs_batch
  
} // namespace sill

#include <sill/macros_undef.hpp>

#include <sill/learning/discriminative/load_functions.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_ALL_PAIRS_BATCH_HPP
