
#ifndef _SILL_MULTICLASS_LOGISTIC_REGRESSION_HPP_
#define _SILL_MULTICLASS_LOGISTIC_REGRESSION_HPP_

#include <algorithm>

#include <sill/functional.hpp>
#include <sill/learning/validation/crossval_parameters.hpp>
#include <sill/learning/validation/model_validation_functor.hpp>
#include <sill/learning/validation/validation_framework.hpp>
#include <sill/learning/dataset/ds_oracle.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/discriminative/multiclass_classifier.hpp>
#include <sill/math/linear_algebra/sparse_linear_algebra.hpp>
#include <sill/math/statistics.hpp>
#include <sill/optimization/logreg_opt_vector.hpp>
#include <sill/optimization/real_optimizer_builder.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  struct multiclass_logistic_regression_parameters {

    // Learning parameters
    //==========================================================================

    //! 0: none
    //! 1: L_1  (lambda * |x|)
    //! 2: L_2  (lambda * .5 * dot(x,x))
    //!  (default = L_2)
    size_t regularization;

    //! Regularization parameter
    //!  (default = .00001)
    double lambda;

    //! Number of initial iterations to run.
    //!  (default = 1000)
    size_t init_iterations;

    //! Range [-perturb,perturb] within
    //! which to choose perturbed values for initial parameters.
    //! Note: This should generally not be used with L1 regularization.
    //!  (default = 0)
    double perturb;

    // Other parameters
    //==========================================================================

    /**
     * If the regression weights become too large, logistic regression can
     * run into numerical problems.  In particular, the exponentiation of the
     * weights * features for each class can be (numerically) infinite.
     * This parameter lets the user specify how to handle the case when
     * prediction methods run into infinities when computing probabilities:
     *  - false: Throw an error.
     *     (default)
     *  - true: If one or more classes' terms are infinite, predict the max
     *          (in log-space) as having probability 1 and the other as 0.
     *          If multiple of these classes' terms are equal in log-space,
     *          give them equal probability and the other classes 0 probability.
     */
    bool resolve_numerical_problems;

    //! Used to make the algorithm deterministic
    //!  (default = time)
    unsigned random_seed;

    /**
     * Print debugging info:
     *  - 0: none (default)
     *  - 1: print warnings
     *  - 2: print extra stuff
     *  - higher: revert to highest debugging mode
     */
    size_t debug;

    // Optimization parameters
    //==========================================================================

    //! Optimization method.
    real_optimizer_builder::real_optimizer_type opt_method;

    //! Gradient method parameters.
    gradient_method_parameters gm_params;

    //! Conjugate gradient update method.
    size_t cg_update_method;

    //! L-BFGS M.
    size_t lbfgs_M;

    // Methods
    //==========================================================================

    multiclass_logistic_regression_parameters();

    bool valid() const;

    void save(std::ofstream& out) const;

    void load(std::ifstream& in);

    void save(oarchive& ar) const;

    void load(iarchive& ar);

    void print(std::ostream& out, const std::string& line_prefix = "") const;

  }; // struct multiclass_logistic_regression_parameters

  std::ostream&
  operator<<(std::ostream& out,
             const multiclass_logistic_regression_parameters& mlr_params);

  /**
   * Class for learning a multiclass logistic regression model.
   *  - The parameters may be set to support different optimization methods
   *    and different types of regularization.
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @todo serialization
   * @todo confidence-rated predictions, regularization, different optimization
   *       objectives
   * @todo Add L-BFGS support.
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   */
  template <typename LA = dense_linear_algebra<> >
  class multiclass_logistic_regression : public multiclass_classifier<LA> {

    // Public types
    //==========================================================================
  public:

    typedef multiclass_classifier<LA> base;

    typedef typename base::la_type            la_type;
    typedef typename base::record_type        record_type;
    typedef typename base::value_type         value_type;
    typedef typename base::vector_type        vector_type;
    typedef typename base::matrix_type        matrix_type;
    typedef typename base::dense_vector_type  dense_vector_type;
    typedef typename base::dense_matrix_type  dense_matrix_type;

    typedef logreg_opt_vector<value_type> opt_variables;

    // Public methods
    //==========================================================================
  public:

    // Virtual methods from base classes (*means pure virtual):
    //  From learner:
    //   name()*            x
    //   is_online()*       x
    //   training_time()     
    //   random_seed()      x
    //   save(), load()
    //  From classifier:
    //   is_confidence_rated()
    //   train_accuracy()
    //   set_confidences()
    //  From singlelabel_classifier:
    //   predict()*
    //  From multiclass_classifier:
    //   create()*           x
    //   confidences()        x
    //   predict_raws()
    //   probabilities()       x

    // Constructors and destructors
    //==========================================================================

    /**
     * Constructor which builds nothing but may be used to create other
     * instances.
     * @param parameters    algorithm parameters
     */
    explicit multiclass_logistic_regression
    (multiclass_logistic_regression_parameters params
     = multiclass_logistic_regression_parameters())
      : params(params), my_ds_ptr(NULL), my_ds_o_ptr(NULL), ds_ptr(NULL),
        o_ptr(NULL) {
      init(true, 0);
    }

    /**
     * Constructor.
     * @param stats         a statistics class for the training dataset
     * @param parameters    algorithm parameters
     */
    explicit multiclass_logistic_regression
    (dataset_statistics<la_type>& stats,
     multiclass_logistic_regression_parameters params
     = multiclass_logistic_regression_parameters())
      : base(stats.get_dataset()), params(params),
        my_ds_ptr(NULL), my_ds_o_ptr(NULL), ds_ptr(&(stats.get_dataset())),
        o_ptr(NULL) {
      init(true, 0);
      build();
    }

    /**
     * Constructor.
     * @param o    data oracle
     * @param n    max number of examples which should be drawn from the oracle
     * @param parameters    algorithm parameters
     */
    multiclass_logistic_regression
    (oracle<la_type>& o, size_t n,
     multiclass_logistic_regression_parameters params
     = multiclass_logistic_regression_parameters())
      : base(o), params(params),
        my_ds_ptr(NULL), my_ds_o_ptr(NULL), ds_ptr(NULL), o_ptr(&o) {
      init(true, n);
      build();
    }

    /**
     * Constructor.
     * @param ds            training dataset
     * @param parameters    algorithm parameters
     */
    multiclass_logistic_regression
    (const dataset<la_type>& ds,
     multiclass_logistic_regression_parameters params
     = multiclass_logistic_regression_parameters())
      : base(ds), params(params),
        my_ds_ptr(NULL), my_ds_o_ptr(NULL), ds_ptr(&ds), o_ptr(NULL) {
      init(true, 0);
      build();
    }

    /**
     * Constructor for warm starts.
     * @param stats         a statistics class for the training dataset
     * @param init_mlr      model used for a warm start
     * @param parameters    algorithm parameters
     */
    multiclass_logistic_regression
    (const dataset<la_type>& ds,
     const multiclass_logistic_regression& init_mlr,
     multiclass_logistic_regression_parameters params
     = multiclass_logistic_regression_parameters())
      : base(ds), params(params),
        my_ds_ptr(NULL), my_ds_o_ptr(NULL), ds_ptr(&ds), o_ptr(NULL),
        weights_(init_mlr.weights_) {
      init(false, 0);
      // Check base:
      assert(label_ == init_mlr.label_);
      assert(label_index_ == init_mlr.label_index_);
      // Check this:
      assert(finite_seq == init_mlr.finite_seq);
      assert(vector_seq == init_mlr.vector_seq);
      assert(nclasses_ == init_mlr.nclasses_);

      params.perturb = 0;
      build();
    }

    ~multiclass_logistic_regression() {
      clear_pointers();
    }

    //! Train a new multiclass classifier of this type with the given data.
    boost::shared_ptr<multiclass_classifier<la_type> >
    create(dataset_statistics<la_type>& stats) const {
      boost::shared_ptr<multiclass_classifier<la_type> >
        bptr(new multiclass_logistic_regression(stats, this->params));
      return bptr;
    }

    //! Train a new multiclass classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    boost::shared_ptr<multiclass_classifier<la_type> >
    create(oracle<la_type>& o, size_t n) const {
      boost::shared_ptr<multiclass_classifier<la_type> >
        bptr(new multiclass_logistic_regression(o, n, this->params));
      return bptr;
    }

    //! WARNING: This does NOT copy optimization info carefully,
    //!          so it may not work to copy this class during optimization.
    multiclass_logistic_regression(const multiclass_logistic_regression& other){
      *this = other;
    }

    //! WARNING: This does NOT copy optimization info carefully,
    //!          so it may not work to copy this class during optimization.
    multiclass_logistic_regression&
    operator=(const multiclass_logistic_regression& other);

    // Getters and helpers
    //==========================================================================

    using base::nclasses;

    //! Return a name for the algorithm without template parameters.
    std::string name() const {
      return "multiclass_logistic_regression";
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

    //! Returns training accuracy (or estimate of it).
    double train_accuracy() const;

    operator std::string() const {
      std::ostringstream out; out << *this; return out.str();
    }

    //! Print classifier
    void print(std::ostream& out) const {
      out << "Multiclass Logistic Regression\n";
      out << "  w (finite) =";
      for (size_t k = 0; k < nclasses_; ++k)
        out << "   (class " << k << "): " << weights_.f.row(k) << "\n";
      out << "\n  w (vector) =";
      for (size_t k = 0; k < nclasses_; ++k)
        out << "   (class " << k << "): " << weights_.v.row(k) << "\n";
      out << "\n  b = " << weights_.b << "\n"
          << " (estimate of) training accuracy = " << train_accuracy() << "\n";
    }

    /**
     * This returns the set of finite input (non-class) variables which have
     * non-zero weights in the learned logistic regressor.
     */
    finite_domain get_finite_dependencies() const {
      finite_domain x;
      for (size_t i = 0; i < finite_seq.size(); ++i) {
        if (sum(abs(weights_.f.col(i))) > params.gm_params.convergence_zero)
          x.insert(finite_seq[i]);
      }
      return x;
    }

    /**
     * This returns the set of vector input (non-class) variables which have
     * non-zero weights in the learned logistic regressor.
     */
    vector_domain get_vector_dependencies() const {
      vector_domain x;
      for (size_t i = 0; i < vector_seq.size(); ++i) {
        if (sum(abs(weights_.v.col(i))) > params.gm_params.convergence_zero)
          x.insert(vector_seq[i]);
      }
      return x;
    }

    /**
     * Informs this classifier that you will give it records of this form
     * (i.e., with the same variables and variable orderings).
     * This helps speed up the classification methods.
     * You MUST call unfix_record() before giving it records of another form.
     */
    void fix_record(const record_type& r);

    /**
     * Undoes fix_record().
     */
    void unfix_record();

    /**
     * If available, this prints out statistics about the optimization done
     * during learning.
     */
    void print_optimization_stats(std::ostream& out) const {
      switch (params.opt_method) {
      case real_optimizer_builder::GRADIENT_DESCENT:
      case real_optimizer_builder::CONJUGATE_GRADIENT:
      case real_optimizer_builder::CONJUGATE_GRADIENT_DIAG_PREC:
      case real_optimizer_builder::LBFGS:
      case real_optimizer_builder::STOCHASTIC_GRADIENT:
        if (optimizer_ptr)
          out << "multiclass_logistic_regression used a gradient_method:\n"
              << "\t iteration = " << optimizer_ptr->iteration() << "\n";
        break;
      default:
        assert(false);
      }
    }

    // Learning and mutating operations
    //==========================================================================

    //! Current training objective
    //! This is currently not set for optimization via stochastic gradient.
    double train_objective() const { return train_obj; }

    //! Reset the random seed in this algorithm's parameters and in its
    //! random number generator.
    void random_seed(double value) {
      params.random_seed = value;
      rng.seed(static_cast<unsigned>(value));
    }

    /**
     * Logistic regression weights.
     * WARNING: Do not use this if you do not know what you are doing!
     *   These are exposed to allow meta-learners (such as a CRF whose
     *   factors are logistic regressors) to control the learning of the
     *   regression weights.  
     */
    const opt_variables& weights() const {
      return weights_;
    }

    /**
     * Logistic regression weights.
     * WARNING: Do not use this if you do not know what you are doing!
     *   These are exposed to allow meta-learners (such as a CRF whose
     *   factors are logistic regressors) to control the learning of the
     *   regression weights.  
     */
    opt_variables& weights() {
      return weights_;
    }

    /**
     * Used by add_gradient from log_reg_crf_factor.
     * @todo Figure out a better way to do this.
     */
    void add_gradient(opt_variables& grad, const assignment& a, double w) const;

    /**
     * Used by add_gradient from log_reg_crf_factor.
     * @todo Figure out a better way to do this.
     */
    void
    add_gradient(opt_variables& grad, const record_type& r, double w) const;

    /**
     * Used by add_expected_gradient from log_reg_crf_factor.
     * @todo Figure out a better way to do this.
     */
    void add_expected_gradient(opt_variables& grad, const assignment& a,
                               const table_factor& fy, double w = 1.) const;

    /**
     * Used by add_expected_gradient from log_reg_crf_factor.
     * @todo Figure out a better way to do this.
     */
    void add_expected_gradient(opt_variables& grad, const record_type& r,
                               const table_factor& fy, double w = 1.) const;

    /**
     * Used by add_combined_gradient from log_reg_crf_factor.
     * @todo Figure out a better way to do this.
     */
    void add_combined_gradient(opt_variables& grad, const record_type& r,
                               const table_factor& fy, double w = 1.) const;

    /**
     * Used by log_reg_crf_factor::add_expected_squared_gradient.
     * @todo Figure out a better way to do this.
     */
    void
    add_expected_squared_gradient(opt_variables& hd, const assignment& a,
                                  const table_factor& fy, double w = 1.) const;

    /**
     * Used by log_reg_crf_factor::add_expected_squared_gradient.
     * @todo Figure out a better way to do this.
     */
    void
    add_expected_squared_gradient(opt_variables& hd, const record_type& r,
                                  const table_factor& fy, double w = 1.) const;

    // Prediction methods
    //==========================================================================

    //! Predict the label of a new example.
    std::size_t predict(const record_type& example) const {
      my_probabilities(example, tmpvec);
      return max_index(tmpvec, rng);
    }

    //! Predict the label of a new example.
    std::size_t predict(const assignment& example) const {
      my_probabilities(example, tmpvec);
      return max_index(tmpvec, rng);
    }

    //! Predict the probability of the class variable having each value.
    dense_vector_type probabilities(const record_type& example) const {
      my_probabilities(example, tmpvec);
      return tmpvec;
    }

    //! Predict the probability of the class variable having each value.
    dense_vector_type probabilities(const assignment& example) const {
      my_probabilities(example, tmpvec);
      return tmpvec;
    }

    // Methods for iterative learners
    //==========================================================================

    //! Returns the current iteration number (from 0)
    //!  (i.e., the number of learning iterations completed).
    //! ITERATIVE ONLY: This must be implemented by iterative learners.
    size_t iteration() const {
      return iteration_;
    }

    /*
    //! Returns the total time elapsed after each iteration.
    //! ITERATIVE ONLY: This may be implemented by iterative learners.
    std::vector<double> elapsed_times() const {
    }
    */

    //! Does the next step of training (updating the current classifier).
    //! @return true iff the learner may be trained further
    //! ITERATIVE ONLY: This must be implemented by iterative learners.
    bool step();

    /*
    //! Resets the data source to be used in future rounds of training.
    //! @param  n   max number of examples which may be drawn from the oracle
    //! ITERATIVE ONLY: This may be implemented by iterative learners.
    void reset_datasource(oracle<la_type>& o, size_t n) { assert(false); }

    //! Resets the data source to be used in future rounds of training.
    //! ITERATIVE ONLY: This may be implemented by iterative learners.
    void reset_datasource(dataset_statistics<la_type>& stats) { assert(false); }
    */

    //! Call this after learning to free memory.
    //! NOTE: Once this method has been called, step() may fail!
    //! ITERATIVE ONLY: This may be implemented by iterative learners.
    void finish_learning() {
      clear_pointers();
    }

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

    // Methods for choosing lambda via cross validation
    //==========================================================================

    /**
     * Choose the regularization parameter lambda via N-fold cross validation.
     *
     * @param reg_params  (Return value.) lambdas which were tried
     * @param means       (Return value.) means[i] = avg score for lambdas[i]
     * @param stderrs     (Return value.) corresponding std errors of scores
     * @param cv_params   Parameters specifying how to do cross validation.
     * @param ds_ptr      Training data.
     * @param params      Parameters for this class.
     * @param random_seed This uses this random seed, not the one in the
     *                    algorithm parameters.
     *
     * @return  chosen lambda
     */
    static double
    choose_lambda
    (std::vector<dense_vector_type>& lambdas,
     dense_vector_type& means, dense_vector_type& stderrs,
     const crossval_parameters& cv_params, const dataset<la_type>& ds,
     const multiclass_logistic_regression_parameters& params,
     unsigned random_seed);

    /**
     * Choose the regularization parameter lambda via N-fold cross validation.
     *
     * @param cv_params   Parameters specifying how to do cross validation.
     * @param ds_ptr      Training data.
     * @param params      Parameters for this class.
     * @param random_seed This uses this random seed, not the one in the
     *                    algorithm parameters.
     *
     * @return  chosen lambda
     */
    static double
    choose_lambda
    (const crossval_parameters& cv_params, const dataset<la_type>& ds,
     const multiclass_logistic_regression_parameters& params,
     unsigned random_seed);

    // Protected types
    //==========================================================================
  protected:

    //! Objective functor usable with optimization routines.
    class objective_functor {

      const multiclass_logistic_regression& mlr;

      mutable typename dataset<la_type>::record_iterator_type ds_it;

      typename dataset<la_type>::record_iterator_type ds_end;

    public:
      objective_functor(const multiclass_logistic_regression& mlr)
        : mlr(mlr), ds_it(mlr.ds_ptr->begin()), ds_end(mlr.ds_ptr->end()) { }

      //! Computes the value of the objective at x.
      double objective(const opt_variables& x) const {
        double neg_ll = 0;
        size_t i = 0;
        ds_it.reset();
        while (ds_it != ds_end) {
          dense_vector_type v;
          mlr.my_probabilities(*ds_it, v, x.f, x.v, x.b);
          const std::vector<size_t>& findata = (*ds_it).finite();
          size_t label_(findata[mlr.label_index_]);
          neg_ll -= mlr.ds_ptr->weight(i) * std::log(v[label_]);
          ++i;
          ++ds_it;
        }
        neg_ll /= mlr.total_train_weight;

        switch(mlr.params.regularization) {
        case 0:
          break;
        case 1:
          neg_ll += mlr.params.lambda * x.L1norm();
          break;
        case 2:
          {
            double tmpval(x.L2norm());
            neg_ll += mlr.params.lambda * .5 * tmpval * tmpval;
          }
          break;
        default:
          assert(false);
        }

        return neg_ll;
      }

    }; // class objective_functor

    //! Gradient functor used with optimization routines.
    struct mlr_gradient_functor {

      mlr_gradient_functor(const multiclass_logistic_regression& mlr,
                           bool stochastic)
        : mlr(mlr), stochastic(stochastic) { }

      //! Computes the gradient of the function at x.
      //! @param grad  Data type in which to store the gradient.
      void add_gradient(opt_variables& grad, const opt_variables& x,
                        double w) const {
        if (stochastic)
          mlr.add_stochastic_gradient(grad, x, w);
        else
          mlr.add_gradient(grad, x, w);
      }

      //! Computes the gradient of the function at x.
      //! @param grad  Location in which to store the gradient.
      void gradient(opt_variables& grad, const opt_variables& x) const {
        grad.zeros();
        add_gradient(grad, x, 1);
      }

    private:
      const multiclass_logistic_regression& mlr;
      bool stochastic;

    }; // struct mlr_gradient_functor

    //! Diagonal preconditioner functor usable with optimization routines.
    //! Fits the PreconditionerFunctor concept.
    class preconditioner_functor {

      const multiclass_logistic_regression& mlr;

      mutable opt_variables preconditioner;

    public:
      preconditioner_functor(const multiclass_logistic_regression& mlr)
        : mlr(mlr) { }

      //! Applies a preconditioner to the given direction,
      //! when the optimization variables have value x.
      void precondition(opt_variables& direction, const opt_variables& x) const{
	mlr.my_hessian_diag(preconditioner, x);
        preconditioner.reciprocal();
	direction.elem_mult(preconditioner);
      }

      //! Applies the last computed preconditioner to the given direction.
      void precondition(opt_variables& direction) const {
	assert(preconditioner.size() == direction.size());
	direction.elem_mult(preconditioner);
      }

    };  // class preconditioner_functor

    //! Struct used for specialized implementations for dense/sparse SGD.
    template <typename LAType>
    struct sgd_specializer {
    }; // struct sgd_specializer

    friend struct sgd_specializer<la_type>;

    template <typename T, typename I>
    struct sgd_specializer<dense_linear_algebra<T,I> > {

      static void
      init_optimization_stochastic(multiclass_logistic_regression& mlr) {
        stochastic_gradient_parameters sg_params(mlr.params.gm_params);
        if (mlr.params.init_iterations != 0)
          sg_params.single_opt_step_params.set_shrink_eta
            (mlr.params.init_iterations);
        sg_params.add_gradient_inplace = true;
        typedef stochastic_gradient<opt_variables,mlr_gradient_functor>
          stochastic_gradient_type;
        mlr.optimizer_ptr =
          new stochastic_gradient_type(*mlr.grad_functor_ptr, mlr.weights_,
                                       sg_params);
      }

      static bool step_stochastic(multiclass_logistic_regression& mlr) {
        assert(mlr.optimizer_ptr);
        if (!mlr.optimizer_ptr->step())
          return false;
        ++mlr.iteration_;
        return true;
      }
    }; // struct sgd_specializer<dense_linear_algebra<T,I> >

    template <typename T, typename I>
    struct sgd_specializer<sparse_linear_algebra<T,I> > {

      static void
      init_optimization_stochastic(multiclass_logistic_regression& mlr) {
        assert(mlr.lambda < 1);
        mlr.ssgd_eta = mlr.params.gm_params.single_opt_step_params.init_eta;
        mlr.ssgd_cumsum_log_etas.push_back
          (std::log(1 - mlr.ssgd_eta * mlr.lambda));
        mlr.ssgd_shrink_eta =
          mlr.params.gm_params.single_opt_step_params.shrink_eta;
        mlr.ssgd_weights_v.zeros(mlr.weights_.v.n_rows, mlr.weights_.v.n_cols);
      }

      static bool step_stochastic(multiclass_logistic_regression& mlr) {
        if (!mlr.o_ptr->next())
          assert(false);
        const record_type& r = mlr.o_ptr->current();

        // Equivalent to these steps:
        //   my_probabilities(r, tmpvec, weights_.f, weights_.v, weights_.b);
        //   add_reg_gradient(weights_, w, weights_);
        mlr.tmpvec = mlr.weights_.b;
        const std::vector<size_t>& findata = r.finite();
        for (size_t k = 0; k < mlr.nclasses_; ++k) {
          for (size_t j = 0; j < mlr.finite_indices.size(); ++j) {
            size_t val = findata[mlr.finite_indices[j]];
            mlr.tmpvec[k] += mlr.weights_.f(k, mlr.finite_offset[j] + val);
          }
        }
        if (mlr.weights_.v.size() != 0) {
          // tmpvec += weights_.v * r.vector();
          for (I i = 0; i < mlr.tmpvec.size(); ++i) {
            T dot_val = 0;
            for (I k = 0; k < r.vector().num_non_zeros(); ++k) {
              I j = r.vector().index(k);
              T& weights_v_i_j = mlr.weights_.v(i,j);
              // Update weights_v_i_j with regularization history.
              weights_v_i_j *=
                std::exp(mlr.ssgd_cumsum_log_etas[mlr.iteration_]
                         - mlr.ssgd_cumsum_log_etas[mlr.ssgd_weights_v(i,j)]);
              mlr.ssgd_weights_v(i,j) = mlr.iteration_;
              // Contribute to dot product.
              dot_val += weights_v_i_j * r.vector().value(k);
            }
            mlr.tmpvec[i] += dot_val;
          }
        }
        mlr.finish_probabilities(mlr.tmpvec);

        // All weights used here will have been updated already.
        double acc = 0;
        double ll = 0;
        mlr.add_raw_gradient(mlr.weights_, acc, ll, r, mlr.o_ptr->weight(),
                             -mlr.ssgd_eta, mlr.tmpvec);

        // Update eta and eta history.
        mlr.ssgd_eta *= mlr.ssgd_shrink_eta;
        mlr.ssgd_cumsum_log_etas.push_back
          (std::log(1 - mlr.ssgd_eta * mlr.lambda));

        ++mlr.iteration_;
        return true;
      }
    }; // struct sgd_specializer<sparse_linear_algebra<T,I> >

    // Protected data members
    //==========================================================================

    using base::label_;
    using base::label_index_;

    multiclass_logistic_regression_parameters params;

    mutable boost::mt11213b rng;

    //! Dataset pointer (for simulating batch learning from an oracle)
    vector_dataset<la_type>* my_ds_ptr;

    //! Oracle pointer (for simulating online learning from a dataset)
    ds_oracle<la_type>* my_ds_o_ptr;

    //! Dataset (for batch learning)
    const dataset<la_type>* ds_ptr;

    //! Oracle (for online learning)
    oracle<la_type>* o_ptr;

    //! datasource.finite_list()
    finite_var_vector finite_seq;

    //! datasource.vector_list()
    vector_var_vector vector_seq;

    //! True if fix_record() has been called.
    bool fixed_record;

    //! List of indices of non-class finite variables
    //! @todo See if it's faster just to compare with the class variable index.
    std::vector<size_t> finite_indices;

    //! finite_offset[j] = first index in w_fin for finite variable j,
    //!  as indexed in finite_indices
    //!  (so finite_offset[j] + k is the index for value k)
    std::vector<size_t> finite_offset;

    //! vector_offset[j] = first index in w_vec for vector variable j
    //!  (so vector_offset[j] + k is the index for value k of variable j)
    std::vector<size_t> vector_offset;

    /*
      This model predicts:
      P(label k | features x) = exp(w_k'x + b_k) / sum_j exp(w_j'x + b_j)
     */

    //! Lambda
    double lambda;

    //! Weights
    opt_variables weights_;

    //! For all generic optimization methods
    objective_functor* obj_functor_ptr;

    //! For all generic optimization methods
    mlr_gradient_functor* grad_functor_ptr;

    //! For all generic optimization methods
    preconditioner_functor* prec_functor_ptr;

    //! For batch and stochastic optimization methods
    real_optimizer<opt_variables>* optimizer_ptr;

    //! Current iteration; i.e., # iterations completed
    size_t iteration_;

    //! Total weight of training examples
    double total_train_weight;

    //! Number of classes
    size_t nclasses_;

    //! Current training accuracy
    double train_acc;

    //! Current training objective
    double train_obj;

    //! Temp vector for making predictions (to avoid reallocation).
    mutable dense_vector_type tmpvec;

    //! Stores std::log(std::numeric_limits<double>::max() / nclasses_).
    double log_max_double;

    //! (For stochastic gradient with sparse data)
    //! ssgd_cumsum_log_etas[iter] = sum_{t=1}^iter  log(1 - eta_t * lambda)
    std::vector<double> ssgd_cumsum_log_etas;

    //! (For stochastic gradient with sparse data)
    //! Current eta.
    double ssgd_eta;

    //! (For stochastic gradient with sparse data)
    //! Copied from params.
    double ssgd_shrink_eta;

    //! (For stochastic gradient with sparse data)
    //! Elements of ssgd_weights_v correspond to weights_.f and mark the
    //! last iteration on which those elements were updated.
    umat ssgd_weights_v;

    // Protected methods
    //==========================================================================

    /**
     * Initializes stuff using preset ds, o.
     * @param init_weights  If true, initialize the optimization variables.
     * @param n             If drawing a dataset from an oracle, draw this many
     *                      samples.
     */
    void init(bool init_weights, size_t n);

    //! Initialize optimization-related pointers (not ds,o pointers).
    void init_optimization();

    //! Specialized for dense and sparse linear algebra.
    void init_optimization_stochastic();

    //! Free all pointers with data owned by this class.
    void clear_pointers();

    //! Learn stuff.
    void build();

    //! Given that the vector v is set to the class weights in log-space,
    //! exponentiate them and normalize them to compute probabilities.
    void finish_probabilities(dense_vector_type& v) const;

    //! Set v to be the predicted class conditional probabilities for the
    //! given example, using the given parameters.
    void my_probabilities(const record_type& example, dense_vector_type& v,
                          const dense_matrix_type& w_fin_,
                          const dense_matrix_type& w_vec_,
                          const dense_vector_type& b_) const;

    //! Set v to be the predicted class conditional probabilities for the
    //! given example, using the given parameters.
    void my_probabilities(const assignment& example, dense_vector_type& v,
                          const dense_matrix_type& w_fin_,
                          const dense_matrix_type& w_vec_,
                          const dense_vector_type& b_) const;

    //! Set v to be the predicted class conditional probabilities for the
    //! given example.
    void
    my_probabilities(const record_type& example, dense_vector_type& v) const;

    //! Set v to be the predicted class conditional probabilities for the
    //! given example.
    void
    my_probabilities(const assignment& example, dense_vector_type& v) const;

    /**
     * Compute the gradient of the unregularized objective at x
     * for the given example, adding it to the given opt_variables.
     * Computes some objectives for little extra cost.
     *
     * @param gradient    (Return value) Pre-allocated value to which to
     *                     add the gradient.
     * @param acc         (Return value) Value to which to add the accuracy
     *                     on the example (0/1).
     * @param ll          (Return value) Value to which to add the example's
     *                     log likelihood.
     * @param example     Example to use to calculate the gradient.
     * @param ex_weight   Example weight.
     * @param alt_weight  Weight by which to multiply the gradient
     *                     (in addition to the example weight).
     *                     This weight does not affect acc,ll.
     * @param probs       Predicted class conditional probabilities at x.
     *                    Note: This method alters the probs vector!
     */
    void
    add_raw_gradient(opt_variables& gradient, double& acc, double& ll,
                     const record_type& example, double ex_weight,
                     double alt_weight, dense_vector_type& probs) const;

    /**
     * Compute the gradient of the regularization term of the objective at x,
     * adding it to the given opt_variables.
     *
     * @param gradient    (Return value) Pre-allocated value to which to
     *                     add the gradient.
     * @param alt_weight  Weight by which to multiply the gradient.
     * @param x           Point at which to calculate the gradient (w.r.t. x).
     */
    void
    add_reg_gradient(opt_variables& gradient, double alt_weight,
                     const opt_variables& x) const;

    //! Compute the gradient at x, storing it in the given opt_variables.
    //! Computes some objectives for little extra cost.
    //! @param gradient  Pre-allocated place to store gradient.
    void add_gradient(opt_variables& gradient, const opt_variables& x,
                      double w) const;

    /**
     * Compute a stochastic estimate of the gradient at x, storing it in
     * the given opt_variables.
     * Computes some objectives for little extra cost.
     * @param gradient  Pre-allocated place to store gradient.
     */
    void
    add_stochastic_gradient(opt_variables& gradient,
                            const opt_variables& x, double w) const;

    //! Compute the diagonal of the Hessian at x, storing it in the given
    //! opt_variables.
    //! @param hd    Pre-allocated place to store the Hessian diagonal.
    void
    my_hessian_diag(opt_variables& hd, const opt_variables& x) const;

    //! Specialized for dense and sparse linear algebra.
    bool step_stochastic();

  }; // class multiclass_logistic_regression



  /**
   * Model validation functor for multiclass_logistic_regression.
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   */
  template <typename LA = dense_linear_algebra<> >
  class mlr_validation_functor
    : public model_validation_functor<LA> {

    // Public types
    // =========================================================================
  public:

    typedef model_validation_functor<LA> base;

    typedef LA la_type;
    typedef typename la_type::value_type value_type;
    typedef arma::Col<value_type>        dense_vector_type;

    // Public data
    // =========================================================================

    //! If true, print out info whenever CV is run.
    //!  (default = false)
    bool verbose_cv;

    //! Output stream to write to when verbose CV is called.
    //! If NULL, use std::cout (default).
    std::ostream* out_ptr;

    // Constructors and destructors
    // =========================================================================

    mlr_validation_functor()
      : verbose_cv(false), out_ptr(NULL), mlr_ptr(NULL), do_cv(false) { }

    //! Constructor for testing with the given parameters.
    explicit mlr_validation_functor
    (const multiclass_logistic_regression_parameters& params)
      : verbose_cv(false), out_ptr(NULL), params(params), mlr_ptr(NULL),
        do_cv(false) { }

    //! Constructor for testing via CV, where each round chooses lambda via
    //! a second level of CV.
    mlr_validation_functor
    (const multiclass_logistic_regression_parameters& params,
     const crossval_parameters& cv_params)
      : verbose_cv(false), out_ptr(NULL), params(params), mlr_ptr(NULL),
        do_cv(true), cv_params(cv_params) { }

    mlr_validation_functor(const mlr_validation_functor& other)
      : verbose_cv(other.verbose_cv), out_ptr(other.out_ptr),
        params(other.params), mlr_ptr(NULL),
        do_cv(other.do_cv), cv_params(other.cv_params) {
      if (other.mlr_ptr)
        mlr_ptr = new multiclass_logistic_regression<la_type>(*(other.mlr_ptr));
    }

    mlr_validation_functor& operator=(const mlr_validation_functor& other) {
      verbose_cv = other.verbose_cv;
      out_ptr = other.out_ptr;
      params = other.params;
      mlr_ptr = NULL;
      if (other.mlr_ptr)
        mlr_ptr = new multiclass_logistic_regression<la_type>(*(other.mlr_ptr));
      do_cv = other.do_cv;
      cv_params = other.cv_params;
      return *this;
    }

    ~mlr_validation_functor() {
      if (mlr_ptr)
        delete(mlr_ptr);
    }

    //! Like a constructor.
    void reset(const multiclass_logistic_regression_parameters& params) {
      verbose_cv = false;
      out_ptr = NULL;
      this->params = params;
      if (mlr_ptr)
        delete(mlr_ptr);
      mlr_ptr = NULL;
      do_cv = false;
    }

    //! Like a constructor.
    void reset(const multiclass_logistic_regression_parameters& params,
               const crossval_parameters& cv_params) {
      verbose_cv = false;
      out_ptr = NULL;
      this->params = params;
      if (mlr_ptr)
        delete(mlr_ptr);
      mlr_ptr = NULL;
      do_cv = true;
      this->cv_params = cv_params;
    }

    // Protected data
    // =========================================================================
  protected:

    using base::result_map_;

    multiclass_logistic_regression_parameters params;

    multiclass_logistic_regression<la_type>* mlr_ptr;

    bool do_cv;

    crossval_parameters cv_params;

    // Protected methods
    // =========================================================================

    void train_model(const dataset<la_type>& ds, unsigned random_seed) {
      if (do_cv) {
        boost::mt11213b rng(random_seed);
        boost::uniform_int<int> unif_int(0,std::numeric_limits<int>::max());

        mlr_validation_functor<la_type> mlr_val_func(params);
        validation_framework<la_type>
          val_frame(ds, cv_params, mlr_val_func, unif_int(rng));
        if (verbose_cv) {
          if (out_ptr)
            val_frame.print(*out_ptr, 1);
          else
            val_frame.print(std::cout, 1);
        }
        assert(val_frame.best_lambdas().size() == 1);
        params.lambda = val_frame.best_lambdas()[0];

        if (mlr_val_func.mlr_ptr) {
          if (mlr_ptr)
            delete(mlr_ptr);
          mlr_ptr = mlr_val_func.mlr_ptr;
          mlr_val_func.mlr_ptr = NULL;
        }
        random_seed = unif_int(rng);
      }

      params.random_seed = random_seed;
      if (mlr_ptr) {
        multiclass_logistic_regression<la_type>* tmp_ptr = mlr_ptr;
        mlr_ptr =
          new multiclass_logistic_regression<la_type>(ds, *tmp_ptr, params);
        delete(tmp_ptr); tmp_ptr = NULL;
        if (mlr_ptr->train_objective() == inf()) {
          // Try without the warm start.
          delete(mlr_ptr);
          mlr_ptr = NULL;
          mlr_ptr = new multiclass_logistic_regression<la_type>(ds, params);
        }
      } else {
        mlr_ptr = new multiclass_logistic_regression<la_type>(ds, params);
      }
      mlr_ptr->finish_learning();
    } // train_model(ds, random_seed)

    void train_model(const dataset<la_type>& ds,
                     const dense_vector_type& validation_params,
                     unsigned random_seed) {
      assert(!do_cv); //This would mean choosing lambda within choosing lambda.
      assert(validation_params.size() == 1);
      params.lambda = validation_params[0];
      train_model(ds, random_seed);
    }

    //! Compute results from model, and store them in result_map_.
    //! @param prefix  Prefix to add to result names.
    //! @return  Main result/score.
    value_type
    add_results(const dataset<la_type>& ds, const std::string& prefix) {
      assert(mlr_ptr);
      value_type ll = ds.expected_value(mlr_ptr->log_likelihood());
      result_map_[prefix + "log likelihood"] = ll;
      result_map_[prefix + "accuracy"] = ds.expected_value(mlr_ptr->accuracy());
      return ll;
    }

  }; // class mlr_validation_functor


  /**
   * Class for parsing command-line options to create
   * a multiclass_logistic_regression instance.
   */
  class multiclass_logistic_regression_builder {

    multiclass_logistic_regression_parameters mlr_params;

    real_optimizer_builder real_opt_builder;

  public:

    multiclass_logistic_regression_builder() { }

    /**
     * Add options to the given Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     */
    void add_options(boost::program_options::options_description& desc,
                     const std::string& desc_prefix = "");

    //! Return the CRF Parameter Learner options specified in this builder.
    const multiclass_logistic_regression_parameters& get_parameters();

  }; // class multiclass_logistic_regression_builder


  //============================================================================
  // Implementations of methods in multiclass_logistic_regression
  //============================================================================


  // Protected methods
  //==========================================================================

  template <typename LA>
  void multiclass_logistic_regression<LA>::init(bool init_weights,
                                                size_t n) {
    rng.seed(static_cast<unsigned>(params.random_seed));

    // Init datasource
    if (ds_ptr || o_ptr) {
      if (real_optimizer_builder::is_stochastic(params.opt_method)) {
        if (!o_ptr) {
          assert(ds_ptr);
          typename ds_oracle<la_type>::parameters dso_params;
          dso_params.randomization_period = 5;
          boost::uniform_int<int> unif_int(0,std::numeric_limits<int>::max());
          dso_params.random_seed = unif_int(rng);
          my_ds_o_ptr = new ds_oracle<la_type>(*ds_ptr);
          o_ptr = my_ds_o_ptr;
        }
      } else {
        if (!ds_ptr) {
          assert(o_ptr);
          my_ds_ptr = new vector_dataset<la_type>(o_ptr->datasource_info(),n);
          for (size_t i = 0; i < n; ++i) {
            if (o_ptr->next())
              my_ds_ptr->insert(o_ptr->current());
            else
              break;
          }
          ds_ptr = my_ds_ptr;
        }
      }
    }

    // Init other stuff
    fixed_record = false;
    if (ds_ptr) {
      finite_seq = ds_ptr->finite_list();
      vector_seq = ds_ptr->vector_list();
      if (init_weights) {
        assert(label_);
        weights_.resize(typename opt_variables::size_type
                        (label_->size(),ds_ptr->finite_dim() - label_->size(),
                         label_->size(), ds_ptr->vector_dim(),
                         label_->size()));
        weights_.zeros();
      }
    }
    obj_functor_ptr = NULL;
    grad_functor_ptr = NULL;
    prec_functor_ptr = NULL;
    optimizer_ptr = NULL;
    iteration_ = 0;
    total_train_weight = 0;
    train_acc = 0;
    train_obj = std::numeric_limits<double>::max();
    if (label_) {
      nclasses_ = label_->size();
      tmpvec.set_size(label_->size());
    }
    if (nclasses_ != 0)
      log_max_double = std::log(std::numeric_limits<double>::max()/nclasses_);
    else
      log_max_double = std::log(std::numeric_limits<double>::max());
  } // init(init_weights, n)

  template <typename LA>
  void multiclass_logistic_regression<LA>::init_optimization() {
    switch(params.opt_method) {
    case real_optimizer_builder::CONJUGATE_GRADIENT_DIAG_PREC:
      prec_functor_ptr = new preconditioner_functor(*this);
    case real_optimizer_builder::GRADIENT_DESCENT:
    case real_optimizer_builder::CONJUGATE_GRADIENT:
    case real_optimizer_builder::LBFGS:
      obj_functor_ptr = new objective_functor(*this);
      grad_functor_ptr = new mlr_gradient_functor(*this, false);
      break;
    case real_optimizer_builder::STOCHASTIC_GRADIENT:
      grad_functor_ptr = new mlr_gradient_functor(*this, true);
      break;
    default:
      assert(false);
    }
    switch (params.opt_method) {
    case real_optimizer_builder::GRADIENT_DESCENT:
      {
        gradient_descent_parameters ga_params(params.gm_params);
        typedef
          gradient_descent<opt_variables,objective_functor,mlr_gradient_functor>
          gradient_descent_type;
        optimizer_ptr =
          new gradient_descent_type(*obj_functor_ptr, *grad_functor_ptr,
                                    weights_, ga_params);
      }
      break;
    case real_optimizer_builder::CONJUGATE_GRADIENT:
      {
        conjugate_gradient_parameters cg_params(params.gm_params);
        cg_params.update_method = params.cg_update_method;
        typedef
          conjugate_gradient<opt_variables,objective_functor,mlr_gradient_functor>
          conjugate_gradient_type;
        optimizer_ptr =
          new conjugate_gradient_type(*obj_functor_ptr, *grad_functor_ptr,
                                      weights_, cg_params);
      }
      break;
    case real_optimizer_builder::CONJUGATE_GRADIENT_DIAG_PREC:
      {
        conjugate_gradient_parameters cg_params(params.gm_params);
        cg_params.update_method = params.cg_update_method;
        typedef conjugate_gradient<opt_variables,objective_functor,
                                   mlr_gradient_functor,preconditioner_functor>
          prec_conjugate_gradient_type;
        optimizer_ptr =
          new prec_conjugate_gradient_type(*obj_functor_ptr, *grad_functor_ptr,
					   *prec_functor_ptr, weights_,
					   cg_params);
      }
      break;
    case real_optimizer_builder::LBFGS:
      {
        lbfgs_parameters lbfgs_params(params.gm_params);
        lbfgs_params.M = params.lbfgs_M;
        typedef lbfgs<opt_variables,objective_functor,mlr_gradient_functor>
          lbfgs_type;
        optimizer_ptr =
          new lbfgs_type(*obj_functor_ptr, *grad_functor_ptr,
                         weights_, lbfgs_params);
      }
      break;
    case real_optimizer_builder::STOCHASTIC_GRADIENT:
      init_optimization_stochastic();
      break;
    default:
      assert(false);
    }
  } // init_optimization

  template <typename LA>
  void multiclass_logistic_regression<LA>::init_optimization_stochastic() {
    sgd_specializer<LA>::init_optimization_stochastic(*this);
  } // init_optimization_stochastic

  template <typename LA>
  void multiclass_logistic_regression<LA>::clear_pointers() {
    if (my_ds_ptr)
      delete(my_ds_ptr);
    my_ds_ptr = NULL;
    if (my_ds_o_ptr)
      delete(my_ds_o_ptr);
    my_ds_o_ptr = NULL;
    ds_ptr = NULL;
    o_ptr = NULL;
    if (obj_functor_ptr)
      delete(obj_functor_ptr);
    obj_functor_ptr = NULL;
    if (grad_functor_ptr)
      delete(grad_functor_ptr);
    grad_functor_ptr = NULL;
    if (prec_functor_ptr)
      delete(prec_functor_ptr);
    prec_functor_ptr = NULL;
    if (optimizer_ptr)
      delete(optimizer_ptr);
    optimizer_ptr = NULL;
  }

  template <typename LA>
  void multiclass_logistic_regression<LA>::build() {
    assert(ds_ptr);
    const dataset<la_type>& ds = *ds_ptr;

    // Process dataset and parameters
    assert(params.valid());
    if (ds.num_finite() > 1) {
      finite_offset.push_back(0);
      for (size_t j = 0; j < ds.num_finite(); ++j)
        if (j != label_index_) {
          finite_indices.push_back(j);
          finite_offset.push_back(finite_offset.back()+finite_seq[j]->size());
        }
      finite_offset.pop_back();
    }
    if (ds.num_vector() > 0) {
      vector_offset.push_back(0);
      for (size_t j = 0; j < ds.num_vector() - 1; ++j)
        vector_offset.push_back(vector_offset.back()+vector_seq[j]->size());
    }
    fixed_record = true;
    lambda = params.lambda;
    if (params.regularization == 1 || params.regularization == 2) {
      if (lambda < 0) {
        std::cerr << "multiclass_logistic_regression was given lambda < 0: "
                  << "lambda = " << lambda << std::endl;
        assert(false);
        return;
      }
    }
    for (size_t i = 0; i < ds.size(); ++i)
      total_train_weight += ds.weight(i);

    // Initialize weights
    if (params.perturb > 0) {
      boost::uniform_real<double> uniform_dist;
      uniform_dist = boost::uniform_real<double>
        (-1 * params.perturb, params.perturb);
      for (size_t i = 0; i < weights_.f.n_rows; ++i)
        for (size_t j = 0; j < weights_.f.n_cols; ++j)
          weights_.f(i,j) = uniform_dist(rng);
      for (size_t i = 0; i < weights_.v.n_rows; ++i)
        for (size_t j = 0; j < weights_.v.n_cols; ++j)
          weights_.v(i,j) = uniform_dist(rng);
      foreach(double& v, weights_.b)
        v = uniform_dist(rng);
    }

    init_optimization();

    while (iteration_ < params.init_iterations) {
      if (!step()) {
        if (params.debug > 1)
          std::cerr << "multiclass_logistic_regression terminated"
                    << " optimization after " << iteration_ << "iterations."
                    << std::endl;
        break;
      }
    }
    fixed_record = false;
    if (iteration_ == params.init_iterations && params.init_iterations > 0)
      if (params.debug > 0)
        std::cerr << "WARNING: multiclass_logistic_regression terminated"
                  << " optimization after init_iterations="
                  << params.init_iterations << " steps! Consider doing more."
                  << std::endl;
  } // end of function build()

  template <typename LA>
  void multiclass_logistic_regression<LA>::
  finish_probabilities(dense_vector_type& v) const {
    if (params.resolve_numerical_problems) {
      for (size_t k = 0; k < nclasses_; ++k) {
        if (v[k] > log_max_double) {
          double maxval(v[max_index(v, rng)]);
          for (size_t k = 0; k < nclasses_; ++k) {
            if (v[k] == maxval)
              v[k] = 1;
            else
              v[k] = 0;
          }
          v /= sum(v);
          return;
        }
      }
      for (size_t k = 0; k < nclasses_; ++k)
        v[k] = exp(v[k]);
      double tmpsum(sum(v));
      if (tmpsum == 0) {
        v.ones();
        v /= nclasses_;
        return;
      }
      v /= tmpsum;
    } else {
      for (size_t k = 0; k < nclasses_; ++k) {
        if (v[k] > log_max_double) {
          throw std::runtime_error
            (std::string("multiclass_logistic_regression") +
             " had overflow when computing probabilities.  To deal with such" +
             " overflows in a hacky (but reasonable) way, use the parameter" +
             " resolve_numerical_problems.");
        }
        v[k] = exp(v[k]);
      }
      if (sum(v) == 0)
        throw std::runtime_error
          (std::string("multiclass_logistic_regression") +
           " got all zeros when computing probabilities.  To deal with such" +
           " issues in a hacky (but reasonable) way, use the parameter" +
           " resolve_numerical_problems.");
      v /= sum(v);
    }
  } // finish_probabilities

  template <typename LA>
  void
  multiclass_logistic_regression<LA>::
  my_probabilities(const record_type& example, dense_vector_type& v,
                   const dense_matrix_type& w_fin_,
                   const dense_matrix_type& w_vec_,
                   const dense_vector_type& b_) const {
    v = b_;
    const std::vector<size_t>& findata = example.finite();
    for (size_t k = 0; k < nclasses_; ++k) {
      for (size_t j = 0; j < finite_indices.size(); ++j) {
        size_t val(findata[finite_indices[j]]);
        v[k] += w_fin_(k, finite_offset[j] + val);
      }
    }
    if (w_vec_.size() != 0)
      sill::gemv('n', 1.0, w_vec_, example.vector(), 1.0, v);
//      v += w_vec_ * example.vector();
    finish_probabilities(v);
  }

  template <typename LA>
  void
  multiclass_logistic_regression<LA>::
  my_probabilities(const assignment& example, dense_vector_type& v,
                   const dense_matrix_type& w_fin_,
                   const dense_matrix_type& w_vec_,
                   const dense_vector_type& b_) const {
    v = b_;
    const finite_assignment& fa = example.finite();
    const vector_assignment& va = example.vector();
    for (size_t k = 0; k < nclasses_; ++k) {
      for (size_t j = 0; j < finite_indices.size(); ++j) {
        size_t val(safe_get(fa, finite_seq[finite_indices[j]]));
        v[k] += w_fin_(k, finite_offset[j] + val);
      }
      for (size_t j = 0; j < vector_seq.size(); ++j) {
        const vec& vecdata = safe_get(va, vector_seq[j]);
        for (size_t j2 = 0; j2 < vector_seq[j]->size(); ++j2) {
          size_t ind(vector_offset[j] + j2);
          v[k] += w_vec_(k,ind) * vecdata[j2];
        }
      }
    }
    finish_probabilities(v);
  }

  template <typename LA>
  void
  multiclass_logistic_regression<LA>::
  my_probabilities(const record_type& example, dense_vector_type& v) const {
    if (fixed_record ||
        ((finite_offset.size() == 0 ||
          example.finite_numbering_ptr->size() == finite_offset.size() + 1) &&
         (vector_offset.size() == 0 ||
          example.vector_numbering_ptr->size() == vector_offset.size())))
      return my_probabilities(example, v, weights_.f, weights_.v, weights_.b);
    else
      return my_probabilities(example.assignment(), v, weights_.f, weights_.v,
                              weights_.b);
  }

  template <typename LA>
  void
  multiclass_logistic_regression<LA>::
  my_probabilities(const assignment& example, dense_vector_type& v) const {
    return my_probabilities(example, v, weights_.f, weights_.v, weights_.b);
  }

  template <typename LA>
  void
  multiclass_logistic_regression<LA>::
  add_raw_gradient(opt_variables& gradient, double& acc, double& ll,
                   const record_type& example, double weight, double alt_weight,
                   dense_vector_type& probs) const {
    const std::vector<size_t>& findata = example.finite();
    size_t label_val = findata[label_index_];
    size_t pred_ = max_index(probs, rng);
    if (label_val == pred_)
      acc += weight;
    ll += weight * std::log(probs[label_val]);
    // Update gradients
    probs[label_val] -= 1;
    probs *= weight * alt_weight;
    for (size_t j = 0; j < finite_indices.size(); ++j) {
      size_t val = finite_offset[j] + findata[finite_indices[j]];
      for (size_t k = 0; k < nclasses_; ++k) {
        gradient.f(k, val) += probs[k];
      }
    }
    if (example.vector().size() != 0)
      gradient.v += outer_product(probs, example.vector());
    gradient.b += probs;
  } // add_raw_gradient()

  template <typename LA>
  void
  multiclass_logistic_regression<LA>::
  add_reg_gradient(opt_variables& gradient, double alt_weight,
                   const opt_variables& x) const {
    double w(alt_weight * lambda);
    switch (params.regularization) {
    case 0: // none
      break;
    case 1: // L-1
      // TODO: Figure out a better way to do this.
      //       (Add more functionality to vector.hpp and matrix.hpp?)
      for (size_t i = 0; i < nclasses_; ++i) {
        for (size_t j = 0; j < x.f.n_cols; ++j) {
          if (x.f(i,j) > 0)
            gradient.f(i,j) += w;
          else if (x.f(i,j) < 0)
            gradient.f(i,j) -= w;
        }
        for (size_t j = 0; j < x.v.n_cols; ++j) {
          if (x.v(i,j) > 0)
            gradient.v(i,j) += w;
          else if (x.v(i,j) < 0)
            gradient.v(i,j) -= w;
        }
        if (x.b[i] > 0)
          gradient.b[i] += w;
        else if (x.b[i] < 0)
          gradient.b[i] -= w;
      }
      break;
    case 2: // L-2
      gradient.f += w * x.f;
      gradient.v += w * x.v;
      gradient.b += w * x.b;
      break;
    default:
      assert(false);
    }
  } // add_reg_gradient()

  template <typename LA>
  void
  multiclass_logistic_regression<LA>::
  add_gradient(opt_variables& gradient, const opt_variables& x, double w) const{
    double train_acc = 0;
    double train_log_like = 0;
    typename dataset<la_type>::record_iterator_type it_end(ds_ptr->end());
    size_t i = 0; // index into dataset
    dense_vector_type probs;
    for (typename dataset<la_type>::record_iterator_type it(ds_ptr->begin());
         it != it_end; ++it) {
      const record_type& r = *it;
      my_probabilities(r, probs, x.f, x.v, x.b);
      add_raw_gradient
        (gradient, train_acc, train_log_like, r, ds_ptr->weight(i),
         w / total_train_weight, probs);
      ++i;
    }

    // Update gradients to account for regularization
    add_reg_gradient(gradient, 1, x);
//    return std::make_pair(train_acc, train_log_like);
  } // end of function my_gradient()

  template <typename LA>
  void
  multiclass_logistic_regression<LA>::
  add_stochastic_gradient(opt_variables& gradient,
                          const opt_variables& x, double w) const {
    double acc = 0;
    double ll = 0;
    if (o_ptr->next()) {
      dense_vector_type probs;
      const record_type& r = o_ptr->current();
      my_probabilities(r, probs, x.f, x.v, x.b);
      add_reg_gradient(gradient, w, x);
      add_raw_gradient(gradient, acc, ll, r, o_ptr->weight(), w, probs);
    } else {
      assert(false);
    }
//    return std::make_pair(acc, ll);
  }

  template <typename LA>
  void
  multiclass_logistic_regression<LA>::my_hessian_diag
  (opt_variables& hd, const opt_variables& x) const {

    if (hd.size() != x.size())
      hd.resize(x.size());
    hd.zeros();
    typename dataset<la_type>::record_iterator_type it_end(ds_ptr->end());
    size_t i = 0; // index into ds
    dense_vector_type v;
    vector_type vecdata;
    for (typename dataset<la_type>::record_iterator_type it(ds_ptr->begin());
         it != it_end; ++it) {
      my_probabilities(*it, v, x.f, x.v, x.b);
      v -= v % v;
      v *= ds_ptr->weight(i);
      const std::vector<size_t>& findata = (*it).finite();
      for (size_t j = 0; j < finite_indices.size(); ++j) {
	size_t val(findata[finite_indices[j]]);
	hd.f.col(finite_offset[j] + val) += v;
      }
      vecdata = (*it).vector() % (*it).vector();
      hd.v += outer_product(v, vecdata);
      hd.b += v;
      ++i;
    }
    hd.f /= total_train_weight;
    hd.v /= total_train_weight;
    hd.b /= total_train_weight;

    switch (params.regularization) {
    case 0:
      break;
    case 1:
      // This is supposed to be 0 (even at the discontinuity), right?
      break;
    case 2:
      hd.f += lambda;
      hd.v += lambda;
      hd.b += lambda;
      break;
    default:
      assert(false);
    }

  } // my_hessian_diag()

  // Constructors, etc.
  //==========================================================================

  template <typename LA>
  multiclass_logistic_regression<LA>&
  multiclass_logistic_regression<LA>::
  operator=(const multiclass_logistic_regression& other) {
    this->clear_pointers();

    base::operator=(other);

    params = other.params;
    rng = other.rng;
    if (other.my_ds_ptr) {
      my_ds_ptr = new vector_dataset<la_type>(*(other.my_ds_ptr));
      ds_ptr = my_ds_ptr;
    } else {
      ds_ptr = other.ds_ptr;
    }
    if (other.my_ds_o_ptr) {
      my_ds_o_ptr = new ds_oracle<la_type>(*(other.my_ds_o_ptr));
      o_ptr = my_ds_o_ptr;
    } else {
      o_ptr = other.o_ptr;
    }
    finite_seq = other.finite_seq;
    vector_seq = other.vector_seq;
    fixed_record = other.fixed_record;
    finite_indices = other.finite_indices;
    finite_offset = other.finite_offset;
    vector_offset = other.vector_offset;
    lambda = other.lambda;
    weights_ = other.weights_;

    // To do eventually: Do a more careful deep copy of optimizer_ptr,
    //  but have it reference the *_functor_ptr copies.
    init_optimization();
    ssgd_eta = other.ssgd_eta;
    ssgd_cumsum_log_etas = other.ssgd_cumsum_log_etas;
    ssgd_shrink_eta = other.ssgd_shrink_eta;
    ssgd_weights_v = other.ssgd_weights_v;

    iteration_ = other.iteration_;
    total_train_weight = other.total_train_weight;
    nclasses_ = other.nclasses_;
    train_acc = other.train_acc;
    train_obj = other.train_obj;
    tmpvec = other.tmpvec;
    log_max_double = other.log_max_double;

    return *this;
  } // operator=

  // Getters and helpers
  //==========================================================================

  template <typename LA>
  double multiclass_logistic_regression<LA>::train_accuracy() const {
    switch(params.opt_method) {
    case real_optimizer_builder::GRADIENT_DESCENT:
    case real_optimizer_builder::CONJUGATE_GRADIENT:
    case real_optimizer_builder::CONJUGATE_GRADIENT_DIAG_PREC:
    case real_optimizer_builder::LBFGS:
      return train_acc;
    case real_optimizer_builder::STOCHASTIC_GRADIENT:
      return (total_train_weight == 0 ? -1 : train_acc / total_train_weight);
    default:
      assert(false);
      return -1;
    }
  }

  template <typename LA>
  void multiclass_logistic_regression<LA>::fix_record(const record_type& r) {
    // Set finite_indices, finite_offset, vector_offset.
    // Set finite_seq, vector_seq?
    assert(false); // TO DO
    fixed_record = true;
  }

  template <typename LA>
  void multiclass_logistic_regression<LA>::unfix_record() {
    fixed_record = false;
  }

  template <typename LA>
  void multiclass_logistic_regression<LA>::add_gradient(opt_variables& grad,
                                                    const assignment& a,
                                                    double w) const {
    const finite_assignment& fa = a.finite();
    const vector_assignment& va = a.vector();
    size_t label_val(safe_get(fa,label_));
    for (size_t j = 0; j < finite_indices.size(); ++j) {
      size_t val(safe_get(fa, finite_seq[finite_indices[j]]));
      grad.f(label_val, finite_offset[j] + val) -= w;
    }
    for (size_t j = 0; j < vector_seq.size(); ++j) {
      const vector_type& vecdata = safe_get(va, vector_seq[j]);
      for (size_t j2 = 0; j2 < vecdata.size(); ++j2) {
        size_t ind(vector_offset[j] + j2);
        grad.v(label_val, ind) -= vecdata[j2] * w;
      }
    }
    grad.b[label_val] -= w;
  }

  template <typename LA>
  void multiclass_logistic_regression<LA>::add_gradient(opt_variables& grad,
                                                    const record_type& r,
                                                    double w) const {
    if ((finite_offset.size() != 0 &&
         r.finite_numbering_ptr->size() != finite_offset.size()) ||
        (vector_offset.size() != 0 &&
         r.vector_numbering_ptr->size() != vector_offset.size())) {
      assert(false); // TO DO
      /*
//      add_gradient(grad, r.assignment(), w);

      size_t label_val = r.finite(label_);
      for (size_t j = 0; j < finite_indices.size(); ++j) {
        size_t val = r.finite(finite_seq[finite_indices[j]]);
        grad.f(label_val, finite_offset[j] + val) -= w;
      }

      grad.v.row(label_val) -= r.vector_values(vector_seq) * w;

      grad.b[label_val] -= w;
      */

    } else {
      const std::vector<size_t>& findata = r.finite();
      size_t label_val(findata[label_index_]);
      for (size_t j = 0; j < finite_indices.size(); ++j) {
        size_t val = findata[finite_indices[j]];
        grad.f(label_val, finite_offset[j] + val) -= w;
      }
//      grad.v.row(label_val) -= r.vector() * w;
      vector_type rvec(r.vector() * w);
      grad.v.row(label_val) -= rvec;

      grad.b[label_val] -= w;
    }
  }

  template <typename LA>
  void
  multiclass_logistic_regression<LA>::add_expected_gradient
  (opt_variables& grad, const assignment& a, const table_factor& fy,
   double w) const {
    // Get marginal over label variable.
    table_factor
      label_marginal(fy.marginal(make_domain<finite_variable>(label_)));
    finite_assignment tmpa;
    dense_vector_type r_vector(grad.v.n_cols);
    vector_assignment2vector(a.vector(), vector_seq, r_vector);
    for (size_t label_val = 0; label_val < nclasses_; ++label_val) {
      tmpa[label_] = label_val;
      double label_prob(label_marginal(tmpa));
      if (label_prob == 0)
        continue;
      label_prob *= w;
      grad.v.row(label_val) -= label_prob * r_vector;
      grad.b[label_val] -= label_prob;
    }
    tmpa = a.finite();
    foreach(const finite_assignment& fa, assignments(fy.arguments())) {
      finite_assignment::const_iterator fa_end(fa.end());
      for (finite_assignment::const_iterator fa_it(fa.begin());
           fa_it != fa_end; ++fa_it)
        tmpa[fa_it->first] = fa_it->second;
      size_t label_val(tmpa[label_]);
      for (size_t j = 0; j < finite_indices.size(); ++j) {
        size_t val(tmpa[finite_seq[finite_indices[j]]]);
        grad.f(label_val, finite_offset[j] + val) -= w * fy(fa);
      }
    }
  }

  template <typename LA>
  void
  multiclass_logistic_regression<LA>::add_expected_gradient
  (opt_variables& grad, const record_type& r, const table_factor& fy,
   double w) const {
    if ((finite_offset.size() != 0 &&
         r.finite_numbering_ptr->size() != finite_offset.size()) ||
        (vector_offset.size() != 0 &&
         r.vector_numbering_ptr->size() != vector_offset.size())) {
      assert(false); // TO DO
//      add_expected_gradient(grad, r.assignment(), fy, w);
    } else {
      // Get marginal over label variable.
      table_factor
        label_marginal(fy.marginal(make_domain<finite_variable>(label_)));
      finite_assignment tmpa;
      for (size_t label_val = 0; label_val < nclasses_; ++label_val) {
        tmpa[label_] = label_val;
        double label_prob(label_marginal(tmpa));
        if (label_prob == 0)
          continue;
        label_prob *= w;
        grad.v.row(label_val) -= label_prob * r.vector();
        grad.b[label_val] -= label_prob;
      }
      tmpa = r.finite_assignment();
      foreach(const finite_assignment& fa, assignments(fy.arguments())) {
        finite_assignment::const_iterator fa_end(fa.end());
        for (finite_assignment::const_iterator fa_it(fa.begin());
             fa_it != fa_end; ++fa_it)
          tmpa[fa_it->first] = fa_it->second;
        size_t label_val(tmpa[label_]);
        for (size_t j = 0; j < finite_indices.size(); ++j) {
          size_t val(tmpa[finite_seq[finite_indices[j]]]);
          grad.f(label_val, finite_offset[j] + val) -= w * fy(fa);
        }
      }
    }
  }

  template <typename LA>
  void
  multiclass_logistic_regression<LA>::add_combined_gradient
  (opt_variables& grad, const record_type& r, const table_factor& fy,
   double w) const {
    if ((finite_offset.size() != 0 &&
         r.finite_numbering_ptr->size() != finite_offset.size()) ||
        (vector_offset.size() != 0 &&
         r.vector_numbering_ptr->size() != vector_offset.size())) {
      add_gradient(grad, r, w);
      add_expected_gradient(grad, r, fy, -w);
    } else {
      // Get marginal over label variable.
      table_factor
        label_marginal(fy.marginal(make_domain<finite_variable>(label_)));
      finite_assignment tmpa;
      const std::vector<size_t>& findata = r.finite();
      for (size_t label_val = 0; label_val < nclasses_; ++label_val) {
        tmpa[label_] = label_val;
        double label_prob(label_marginal(tmpa));
        if (label_prob == 0)
          continue;
        if (label_val == findata[label_index_])
          label_prob -= 1.;
        label_prob *= -w;
        grad.v.row(label_val) -= label_prob * r.vector();
        grad.b[label_val] -= label_prob;
      }
      for (size_t j = 0; j < finite_indices.size(); ++j) {
        size_t val(findata[finite_indices[j]]);
        grad.f(findata[label_index_], finite_offset[j] + val) -= w;
      }
      tmpa = r.finite_assignment();
      foreach(const finite_assignment& fa, assignments(fy.arguments())) {
        finite_assignment::const_iterator fa_end(fa.end());
        for (finite_assignment::const_iterator fa_it(fa.begin());
             fa_it != fa_end; ++fa_it)
          tmpa[fa_it->first] = fa_it->second;
        size_t label_val(tmpa[label_]);
        for (size_t j = 0; j < finite_indices.size(); ++j) {
          size_t val(tmpa[finite_seq[finite_indices[j]]]);
          grad.f(label_val, finite_offset[j] + val) += w * fy(fa);
        }
      }
    }
  }

  template <typename LA>
  void multiclass_logistic_regression<LA>::add_expected_squared_gradient
  (opt_variables& hd, const assignment& a, const table_factor& fy,
   double w) const {
    assert(false); // TO DO
  }

  template <typename LA>
  void multiclass_logistic_regression<LA>::add_expected_squared_gradient
  (opt_variables& hd, const record_type& r, const table_factor& fy,
   double w) const {
    if ((finite_offset.size() != 0 &&
         r.finite_numbering_ptr->size() != finite_offset.size()) ||
        (vector_offset.size() != 0 &&
         r.vector_numbering_ptr->size() != vector_offset.size())) {
      add_expected_squared_gradient(hd, r.assignment(), fy, w);
    } else {
      // Get marginal over label variable.
      table_factor
        label_marginal(fy.marginal(make_domain<finite_variable>(label_)));
      finite_assignment tmpa;
      for (size_t label_val = 0; label_val < nclasses_; ++label_val) {
        tmpa[label_] = label_val;
        double label_prob(label_marginal(tmpa));
        if (label_prob == 0)
          continue;
        label_prob *= w;
        hd.v.row(label_val) -= label_prob * (r.vector() % r.vector());
        hd.b[label_val] -= label_prob;
      }
      tmpa = r.finite_assignment();
      foreach(const finite_assignment& fa, assignments(fy.arguments())) {
        finite_assignment::const_iterator fa_end(fa.end());
        for (finite_assignment::const_iterator fa_it(fa.begin());
             fa_it != fa_end; ++fa_it)
          tmpa[fa_it->first] = fa_it->second;
        size_t label_val(tmpa[label_]);
        for (size_t j = 0; j < finite_indices.size(); ++j) {
          size_t val(tmpa[finite_seq[finite_indices[j]]]);
          hd.f(label_val, finite_offset[j] + val) -= w * fy(fa);
        }
      }
    }
  }

  // Methods for iterative learners
  //==========================================================================

  template <typename LA>
  bool multiclass_logistic_regression<LA>::step() {
    if (real_optimizer_builder::is_stochastic(params.opt_method))
      return step_stochastic();

    assert(optimizer_ptr);
    if (!optimizer_ptr->step())
      return false;
    double prev_train_obj(train_obj);
    train_obj = optimizer_ptr->objective();
    if (train_obj == inf()) {
      if (params.debug > 0)
        std::cerr << "multiclass_logistic_regression::step() failed since"
                  << " objective = inf." << std::endl;
      return false;
    }
    if (params.debug > 0) {
      if (train_obj > prev_train_obj) {
        std::cerr << "multiclass_logistic_regression took a step which "
                  << "increased the objective from " << prev_train_obj
                  << " to " << train_obj << std::endl;
      }
      if (params.debug > 1) {
        std::cerr << "change in objective = "
                  << (train_obj - prev_train_obj) << std::endl;
      }
    }
    // Check for convergence
    if (fabs(train_obj - prev_train_obj)
        < params.gm_params.convergence_zero) {
      if (params.debug > 1)
        std::cerr << "multiclass_logistic_regression converged:"
                  << " objective changed from "
                  << prev_train_obj << " to " << train_obj
                  << "; exiting early (iteration " << iteration() << ")."
                  << std::endl;
      return false;
    }
    ++iteration_;
    return true;
  } // step

  template <typename LA>
  bool multiclass_logistic_regression<LA>::step_stochastic() {
    return sgd_specializer<LA>::step_stochastic(*this);
  }

  // Save and load methods
  //==========================================================================

  template <typename LA>
  void multiclass_logistic_regression<LA>::save(std::ofstream& out,
                                            size_t save_part,
                                            bool save_name) const {
    assert(false);
    // TO DO: SAVE POINTERS TO STUFF, ETC.
    base::save(out, save_part, save_name);
    params.save(out);
    out << train_acc << " " << train_obj
        << " " << iteration_ << " " << total_train_weight << "\n";
    for (size_t i = 0; i < weights_.f.n_rows; ++i)
      out << weights_.f.row(i) << " ";
    out << "\n";
    for (size_t i = 0; i < weights_.f.n_rows; ++i)
      out << weights_.v.row(i) << " ";
    out << "\n" << weights_.b << "\n";
  }

  template <typename LA>
  bool multiclass_logistic_regression<LA>::load(std::ifstream& in,
                                            const datasource& ds,
                                            size_t load_part) {
    assert(false);
    // TO DO: CLEAR POINTERS TO STUFF, ETC.
    ds_ptr = NULL;
    if (!(base::load(in, ds, load_part)))
      return false;
    finite_seq = ds.finite_list();
    vector_seq = ds.vector_list();
    finite_offset.clear();
    finite_indices.clear();
    if (ds.num_finite() > 1) {
      finite_offset.push_back(0);
      for (size_t j = 0; j < ds.num_finite(); ++j)
        if (j != label_index_) {
          finite_indices.push_back(j);
          finite_offset.push_back(finite_offset.back()+finite_seq[j]->size());
        }
      finite_offset.pop_back();
    }
    vector_offset.clear();
    if (ds.num_vector() > 0) {
      vector_offset.push_back(0);
      for (size_t j = 0; j < ds.num_vector() - 1; ++j)
        vector_offset.push_back(vector_offset.back()+vector_seq[j]->size());
    }
    params.load(in);
    rng.seed(static_cast<unsigned>(params.random_seed));
    lambda = params.lambda;
    nclasses_ = nclasses();
    std::string line;
    getline(in, line);
    std::istringstream is(line);
    if (!(is >> train_acc))
      assert(false);
    if (!(is >> train_obj))
      assert(false);
    if (!(is >> iteration_))
      assert(false);
    if (!(is >> total_train_weight))
      assert(false);
    getline(in, line);
    is.clear();
    is.str(line);
    weights_.f.set_size(nclasses_, ds.finite_dim() - nclasses_);
    weights_.v.set_size(nclasses_, ds.vector_dim());
    for (size_t j = 0; j < nclasses_; ++j) {
      read_vec(is, tmpvec);
      weights_.f.row(j) = tmpvec;
    }
    getline(in, line);
    is.clear();
    is.str(line);
    for (size_t j = 0; j < nclasses_; ++j) {
      read_vec(is, tmpvec);
      weights_.v.row(j) = tmpvec;
    }
    getline(in, line);
    is.clear();
    is.str(line);
    read_vec(is, weights_.b);
    return true;
  }

  // Methods for choosing lambda via cross validation
  //==========================================================================

  template <typename LA>
  double multiclass_logistic_regression<LA>::
  choose_lambda(std::vector<dense_vector_type>& lambdas,
                dense_vector_type& means, dense_vector_type& stderrs,
                const crossval_parameters& cv_params,
                const dataset<la_type>& ds,
                const multiclass_logistic_regression_parameters& params,
                unsigned random_seed) {
    lambdas.clear(); means.clear(); stderrs.clear();
    mlr_validation_functor<la_type> mlr_val_func(params);
    validation_framework<la_type>
      val_frame(ds, cv_params, mlr_val_func, random_seed);
    assert(val_frame.best_lambdas().size() == 1);
    return val_frame.best_lambdas()[0];
  }

  template <typename LA>
  double multiclass_logistic_regression<LA>::
  choose_lambda(const crossval_parameters& cv_params,
                const dataset<la_type>& ds,
                const multiclass_logistic_regression_parameters& params,
                unsigned random_seed) {
    mlr_validation_functor<la_type> mlr_val_func(params);
    validation_framework<la_type>
      val_frame(ds, cv_params, mlr_val_func, random_seed);
    assert(val_frame.best_lambdas().size() == 1);
    return val_frame.best_lambdas()[0];
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef _SILL_MULTICLASS_LOGISTIC_REGRESSION_HPP_
