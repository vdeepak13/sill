
#ifndef PRL_LEARNING_DISCRIMINATIVE_MULTICLASS_LOGISTIC_REGRESSION_HPP
#define PRL_LEARNING_DISCRIMINATIVE_MULTICLASS_LOGISTIC_REGRESSION_HPP

#include <algorithm>

#include <prl/functional.hpp>
#include <prl/learning/crossval_parameters.hpp>
#include <prl/learning/dataset/ds_oracle.hpp>
#include <prl/learning/dataset/vector_dataset.hpp>
#include <prl/learning/discriminative/multiclass_classifier.hpp>
#include <prl/learning/discriminative/free_functions.hpp>
#include <prl/math/linear_algebra.hpp>
#include <prl/math/statistics.hpp>
#include <prl/optimization/conjugate_gradient.hpp>
#include <prl/optimization/gradient_descent.hpp>
#include <prl/stl_io.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  struct multiclass_logistic_regression_parameters {

    //! Number of initial iterations to run.
    //!  (default = 1000)
    size_t init_iterations;

    //! 0: none
    //! 1: L_1  (lambda * |x|)
    //! 2: L_2  (lambda * .5 * dot(x,x))
    //!  (default = L_2)
    size_t regularization;

    //! Regularization parameter
    //!  (default = .00001)
    double lambda;

    /**
     * 0 = batch gradient descent with line search,
     * 1 = batch conjugate gradient,
     * 2 = stochastic gradient descent (without line search)
     *  (default = 2)
     * 3 = batch conjugate gradient with a diagonal preconditioner
     */
    size_t method;

    //! Learning rate in (0,1]
    //! This is only used to choose the initial step size
    //! if using a line search.
    //!  (default = .1)
    double eta;

    /**
     * Rate at which to decrease the learning rate
     * (by multiplying ETA by MU each round).
     * This is not used if using a line search.
     *  (default = exp(log(10^-4) / INIT_ITERATIONS),
     *            or .999 if INIT_ITERATIONS == 0)
     */
    double mu;

    //! Range [-PERTURB_INIT,PERTURB_INIT] within
    //! which to choose perturbed values for initial parameters.
    //! Note: This should generally not be used with L1 regularization.
    //!  (default = 0)
    double perturb_init;

    //! Amount of change in average log likelihood
    //! below which algorithm will consider itself converged
    //!  (default = .000001)
    double convergence_zero;

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
    double random_seed;

    /**
     * Print debugging info:
     *  - 0: none (default)
     *  - 1: print warnings
     *  - 2: print extra stuff
     *  - higher: revert to highest debugging mode
     */
    size_t debug;

    double choose_mu() const {
      if (init_iterations == 0)
        return .999;
      else
        return exp(-4. * std::log(10.) / init_iterations);
    }

    multiclass_logistic_regression_parameters()
      : init_iterations(1000), regularization(2), lambda(.00001), method(2),
        eta(.1), mu(choose_mu()), perturb_init(0),
        convergence_zero(.000001), resolve_numerical_problems(false),
        random_seed(time(NULL)), debug(0) { }

    bool valid() const {
      if (regularization > 2)
        return false;
      if (lambda < 0)
        return false;
      if (method > 3)
        return false;
      if (eta <= 0 || eta > 1)
        return false;
      if (mu <= 0 || mu > 1)
        return false;
      if (perturb_init < 0)
        return false;
      if (convergence_zero < 0)
        return false;
      return true;
    }

    void save(std::ofstream& out) const {
      out << init_iterations << " " << regularization << " "
          << lambda << " " << method
          << " " << eta
          << " " << mu << " " << perturb_init << " " << convergence_zero
          << " " << (resolve_numerical_problems ? 1 : 0) << " " << random_seed
          << " " << debug << "\n";
    }

    void load(std::ifstream& in) {
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> init_iterations))
        assert(false);
      if (!(is >> regularization))
        assert(false);
      if (!(is >> lambda))
        assert(false);
      if (!(is >> method))
        assert(false);
      if (!(is >> eta))
        assert(false);
      if (!(is >> mu))
        assert(false);
      if (!(is >> perturb_init))
        assert(false);
      if (!(is >> convergence_zero))
        assert(false);
      size_t tmpsize;
      if (!(is >> tmpsize))
        assert(false);
      if (tmpsize == 0)
        resolve_numerical_problems = false;
      else if (tmpsize == 1)
        resolve_numerical_problems = true;
      else
        assert(false);
      if (!(is >> random_seed))
        assert(false);
      if (!(is >> debug))
        assert(false);
    }

  }; // struct multiclass_logistic_regression_parameters

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
   */
  class multiclass_logistic_regression : public multiclass_classifier {

    // Public types
    //==========================================================================
  public:

    typedef multiclass_classifier base;

    /**
     * Optimization variables (which fit the OptimizationVector concept).
     * These are the logistic regression weights.
     */
    struct opt_variables {

      // Types and data
      //------------------------------------------------------------------------

      struct size_type {
        size_t f_rows;
        size_t f_cols;
        size_t v_rows;
        size_t v_cols;
        size_t b_size;

        size_type() { }

        size_type(size_t f_rows, size_t f_cols, size_t v_rows, size_t v_cols,
                  size_t b_size)
          : f_rows(f_rows), f_cols(f_cols), v_rows(v_rows), v_cols(v_cols),
            b_size(b_size) { }

        bool operator==(const size_type& other) const {
          return ((f_rows == other.f_rows) && (f_cols == other.f_cols) &&
                  (v_rows == other.v_rows) && (v_cols == other.v_cols) &&
                  (b_size == other.b_size));
        }

        bool operator!=(const size_type& other) const {
          return (!operator==(other));
        }
      };

      //! f(k,j) = weight for label k, finite index j
      mat f;

      //! v(k,j) = weight for label k, vector index j
      mat v;

      //! Offsets b; b(k) = offset for label k
      vec b;

      // Constructors
      //------------------------------------------------------------------------

      opt_variables() { }

      opt_variables(size_type s, double default_val = 0)
        : f(s.f_rows, s.f_cols, default_val),
          v(s.v_rows, s.v_cols, default_val),
          b(s.b_size, default_val) { }

      opt_variables(const mat& f, const mat& v, const vec& b)
        : f(f), v(v), b(b) { }

      // Getters and non-math setters
      //------------------------------------------------------------------------

      //! Returns true iff this instance equals the other.
      bool operator==(const opt_variables& other) const {
        if ((f != other.f) || (v != other.v) || (b != other.b))
          return false;
        return true;
      }

      //! Returns false iff this instance equals the other.
      bool operator!=(const opt_variables& other) const {
        return !operator==(other);
      }

      size_type size() const {
        return size_type(f.size1(), f.size2(), v.size1(), v.size2(), b.size());
      }

      //! Resize the data.
      void resize(const size_type& newsize) {
        f.resize(newsize.f_rows, newsize.f_cols);
        v.resize(newsize.v_rows, newsize.v_cols);
        b.resize(newsize.b_size);
      }

      // Math operations
      //------------------------------------------------------------------------

      //! Sets all elements to this value.
      opt_variables& operator=(double d) {
        f = d;
        v = d;
        b = d;
        return *this;        
      }

      //! Addition.
      opt_variables operator+(const opt_variables& other) const {
        return opt_variables(f + other.f, v + other.v, b + other.b);
      }

      //! Addition.
      opt_variables& operator+=(const opt_variables& other) {
        f += other.f;
        v += other.v;
        b += other.b;
        return *this;
      }

      //! Subtraction.
      opt_variables operator-(const opt_variables& other) const {
        return opt_variables(f - other.f, v - other.v, b - other.b);
      }

      //! Subtraction.
      opt_variables& operator-=(const opt_variables& other) {
        f -= other.f;
        v -= other.v;
        b -= other.b;
        return *this;
      }

      //! Scalar subtraction.
      opt_variables operator-(double d) const {
        return opt_variables(f - d, v - d, b - d);
      }

      //! Subtraction.
      opt_variables& operator-=(double d) {
        f -= d;
        v -= d;
        b -= d;
        return *this;
      }

      //! Multiplication by a scalar value.
      opt_variables operator*(double d) const {
        return opt_variables(f * d, v * d, b * d);
      }

      //! Multiplication by a scalar value.
      opt_variables& operator*=(double d) {
        f *= d;
        v *= d;
        b *= d;
        return *this;
      }

      //! Division by a scalar value.
      opt_variables operator/(double d) const {
        assert(d != 0);
        return opt_variables(f / d, v / d, b / d);
      }

      //! Division by a scalar value.
      opt_variables& operator/=(double d) {
        assert(d != 0);
        f /= d;
        v /= d;
        b /= d;
        return *this;
      }

      //! Inner product with a value of the same size.
      double inner_prod(const opt_variables& other) const {
        return (elem_mult_sum(f, other.f)
                + elem_mult_sum(v, other.v)
                + prl::inner_prod(b, other.b));
      }

      //! Element-wise multiplication with another value of the same size.
      opt_variables& elem_mult(const opt_variables& other) {
        elem_mult_inplace(other.f, f);
        elem_mult_inplace(other.v, v);
        elem_mult_inplace(other.b, b);
        return *this;
      }

      //! Element-wise reciprocal (i.e., change v to 1/v).
      opt_variables& reciprocal() {
        for (size_t i(0); i < f.size1(); ++i) {
          for (size_t j(0); j < f.size2(); ++j) {
            double& val = f(i,j);
            assert(val != 0);
            val = 1. / val;
          }
        }
        for (size_t i(0); i < v.size1(); ++i) {
          for (size_t j(0); j < v.size2(); ++j) {
            double& val = v(i,j);
            assert(val != 0);
            val = 1. / val;
          }
        }
        for (size_t i(0); i < b.size(); ++i) {
          double& val = b(i);
          assert(val != 0);
          val = 1. / val;
        }
        return *this;
      }

      //! Returns the L1 norm.
      double L1norm() const {
        double l1val(0);
        for (size_t i(0); i < f.size(); ++i)
          l1val += fabs(f(i));
        for (size_t i(0); i < v.size(); ++i)
          l1val += fabs(v(i));
        foreach(double val, b)
          l1val += fabs(val);
        return l1val;
      }

      //! Returns the L2 norm.
      double L2norm() const {
        return sqrt(inner_prod(*this));
      }

      //! Returns a struct of the same size but with values replaced by their
      //! signs (-1 for negative, 0 for 0, 1 for positive).
      opt_variables sign() const {
        opt_variables ov(*this);
        for (size_t i(0); i < f.size(); ++i)
          ov.f(i) = (f(i) > 0 ? 1 : (f(i) == 0 ? 0 : -1) );
        for (size_t i(0); i < v.size(); ++i)
          ov.v(i) = (v(i) > 0 ? 1 : (v(i) == 0 ? 0 : -1) );
        foreach(double& val, ov.b)
          val = (val > 0 ? 1 : (val == 0 ? 0 : -1) );
        return ov;
      }

      //! Sets all values to 0.
      void zeros() {
//        this->operator=(0.);
        f.zeros_memset();
        v.zeros_memset();
        b.zeros_memset();
      }

      //! Print info about this vector (for debugging).
      void print_info(std::ostream& out) const {
        out << "f.size: [" << f.size1() << ", " << f.size2() << "], "
            << "v.size: [" << v.size1() << ", " << v.size2() << "], "
            << "b.size: " << b.size() << "\n";
      }

      //! Print info about extrema in this vector (for debugging).
      void print_extrema_info(std::ostream& out) const {
        for (size_t i(0); i < f.size1(); ++i) {
          if (f.size() > 0)
            out << "f(" << i << ",min) = " << f(i, min_index(f.row(i)))
                << "\t"
                << "f(" << i << ",max) = " << f(i,max_index(f.row(i)))
                << std::endl;
          if (v.size() > 0)
            out << "v(" << i << ",min) = " << v(i, min_index(v.row(i)))
                << "\t"
                << "v(" << i << ",max) = " << v(i, max_index(v.row(i)))
                << std::endl;
        }
        if (b.size() > 0)
          out << "b(min) = " << b(min_index(b)) << "\t"
              << "b(max) = " << b(max_index(b)) << std::endl;
      }

    }; // struct opt_variables

    // Protected types
    //==========================================================================
  protected:

    // Data from base class:
    //  finite_variable* label_
    //  size_t label_index_

    //! Objective functor usable with optimization routines.
    class objective_functor {

      const multiclass_logistic_regression& mlr;

      mutable dataset::record_iterator ds_it;

      dataset::record_iterator ds_end;

    public:
      objective_functor(const multiclass_logistic_regression& mlr)
        : mlr(mlr), ds_it(mlr.ds.begin()), ds_end(mlr.ds.end()) { }

      //! Computes the value of the objective at x.
      double objective(const opt_variables& x) const {
        double ll(0);
        size_t i(0);
        ds_it.reset();
        while (ds_it != ds_end) {
          vec v;
          mlr.my_probabilities(*ds_it, v, x.f, x.v, x.b);
          const std::vector<size_t>& findata = (*ds_it).finite();
          size_t label_(findata[mlr.label_index_]);
          ll -= mlr.ds.weight(i) * std::log(v[label_]);
          ++i;
          ++ds_it;
        }
        ll /= mlr.total_train_weight;

        switch(mlr.params.regularization) {
        case 0:
          break;
        case 1:
          ll += mlr.params.lambda * x.L1norm();
          break;
        case 2:
          {
            double tmpval(x.L2norm());
            ll += mlr.params.lambda * .5 * tmpval * tmpval;
          }
          break;
        default:
          assert(false);
        }

        return ll;
      }

    }; // class objective_functor

    //! Gradient functor usage with optimization routines.
    class gradient_functor {

      const multiclass_logistic_regression& mlr;

    public:
      gradient_functor(const multiclass_logistic_regression& mlr)
        : mlr(mlr) { }

      //! Computes the gradient of the function at x.
      //! @param grad  Data type in which to store the gradient.
      void gradient(opt_variables& grad, const opt_variables& x) const {
        mlr.my_gradient(grad, x);
      }

    }; // class gradient_functor

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

    //! Types for optimization methods
    typedef gradient_method<opt_variables,objective_functor,gradient_functor> gradient_method_type;
    typedef gradient_descent<opt_variables,objective_functor,gradient_functor> gradient_descent_type;
    typedef conjugate_gradient<opt_variables,objective_functor,gradient_functor> conjugate_gradient_type;
    typedef conjugate_gradient<opt_variables,objective_functor,gradient_functor,preconditioner_functor> prec_conjugate_gradient_type;

    //! Helper functor for choose_lambda()
    struct choose_lambda_helper {

      boost::shared_ptr<dataset> ds_ptr;

      const multiclass_logistic_regression_parameters& params_;

      choose_lambda_helper
      (boost::shared_ptr<dataset> ds_ptr,
       const multiclass_logistic_regression_parameters& params_)
        : ds_ptr(ds_ptr), params_(params_) { }

      vec operator()
      (vec& means, vec& stderrs, const std::vector<vec>& lambdas, size_t nfolds,
       unsigned random_seed) const;

    }; // struct choose_lambda_helper

    // Protected data members
    //==========================================================================

    multiclass_logistic_regression_parameters params;

    mutable boost::mt11213b rng;

    //! Dataset pointer (for simulating batch learning from an oracle)
    vector_dataset* ds_ptr;

    //! Oracle pointer (for simulating online learning from a dataset)
    ds_oracle* ds_o_ptr;

    //! Dataset (for batch learning)
    const dataset& ds;

    //! Oracle (for online/stochastic learning)
    oracle& o;

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
    gradient_functor* grad_functor_ptr;

    //! For all generic optimization methods
    preconditioner_functor* prec_functor_ptr;

    //! For batch gradient methods (stores weights)
    gradient_method_type* gradient_method_ptr;

    //! For stochastic gradient descent: gradients (to avoid reallocation)
    opt_variables* gradient_ptr;

    //! For stochastic gradient descent: current eta (learning rate)
    double eta;

    //! Current iteration; i.e., # iterations completed
    size_t iteration_;

    //! Total weight of training examples
    double total_train_weight;

    //! Number of classes
    size_t nclasses_;

    //! Current training accuracy
    double train_acc;

    //! Current training log likelihood
    double train_log_like;

    //! Temp vector for making predictions (to avoid reallocation).
    mutable vec tmpvec;

    //! Stores std::log(std::numeric_limits<double>::max() / nclasses_).
    double log_max_double;

    // Protected methods
    //==========================================================================

    //! Learn stuff.
    void build();

    //! Given that the vector v is set to the class weights in log-space,
    //! exponentiate them and normalize them to compute probabilities.
    void finish_probabilities(vec& v) const;

    //! Set v to be the predicted class conditional probabilities for the
    //! given example, using the given parameters.
    void my_probabilities(const record& example, vec& v,
                          const mat& w_fin_, const mat& w_vec_,
                          const vec& b_) const;

    //! Set v to be the predicted class conditional probabilities for the
    //! given example, using the given parameters.
    void my_probabilities(const assignment& example, vec& v,
                          const mat& w_fin_, const mat& w_vec_,
                          const vec& b_) const;

    //! Set v to be the predicted class conditional probabilities for the
    //! given example.
    void my_probabilities(const record& example, vec& v) const;

    //! Set v to be the predicted class conditional probabilities for the
    //! given example.
    void my_probabilities(const assignment& example, vec& v) const;

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
     * @param x           Point at which to calculate the gradient (w.r.t. x).
     */
    void
    add_raw_gradient(opt_variables& gradient, double& acc, double& ll,
                     const record& example, double ex_weight, double alt_weight,
                     const opt_variables& x) const;

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
    //! @return <avg training accuracy, avg training log likelihood>
    std::pair<double,double>
    my_gradient(opt_variables& gradient, const opt_variables& x) const;

    //! Compute the diagonal of the Hessian at x, storing it in the given
    //! opt_variables.
    //! @param hd    Pre-allocated place to store the Hessian diagonal.
    void
    my_hessian_diag(opt_variables& hd, const opt_variables& x) const;

    //! Take one step of a batch gradient method.
    //! @return  true iff learner may be trained further (and has not converged)
    bool step_gradient_method();

    //! Take one step of stochastic gradient descent.
    //! @todo How should we estimate convergence?
    //!       (Estimate change based on a certain number of previous examples?)
    //! @return  true iff learner may be trained further (and has not converged)
    bool step_stochastic_gradient_descent();

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
      : params(params), ds_ptr(new vector_dataset()),
        ds_o_ptr(new ds_oracle(*ds_ptr)), ds(*ds_ptr), o(*ds_o_ptr),
        fixed_record(false),
        obj_functor_ptr(NULL), grad_functor_ptr(NULL), prec_functor_ptr(NULL),
        gradient_method_ptr(NULL), gradient_ptr(NULL),
        train_acc(-1), train_log_like(-std::numeric_limits<double>::max()),
        log_max_double(std::log(std::numeric_limits<double>::max())) { }

    /**
     * Constructor.
     * @param stats         a statistics class for the training dataset
     * @param parameters    algorithm parameters
     */
    explicit multiclass_logistic_regression
    (statistics& stats,
     multiclass_logistic_regression_parameters params
     = multiclass_logistic_regression_parameters())
      : base(stats.get_dataset()), params(params),
        ds_ptr(NULL), ds_o_ptr(new ds_oracle(stats.get_dataset())),
        ds(stats.get_dataset()), o(*ds_o_ptr),
        finite_seq(stats.get_dataset().finite_list()),
        vector_seq(stats.get_dataset().vector_list()),
        fixed_record(false),
        weights_(opt_variables::size_type
                 (label_->size(),
                  stats.get_dataset().finite_dim() - label_->size(),
                  label_->size(), stats.get_dataset().vector_dim(),
                  label_->size()),
                 0),
        obj_functor_ptr(NULL), grad_functor_ptr(NULL), prec_functor_ptr(NULL),
        gradient_method_ptr(NULL), gradient_ptr(NULL), iteration_(0),
        total_train_weight(0), nclasses_(label_->size()),
        train_acc(0), train_log_like(-std::numeric_limits<double>::max()),
        tmpvec(label_->size()),
        log_max_double(std::log(std::numeric_limits<double>::max()/nclasses_)) {
      build();
    }

    /**
     * Constructor.
     * @param o    data oracle
     * @param n    max number of examples which should be drawn from the oracle
     * @param parameters    algorithm parameters
     */
    multiclass_logistic_regression
    (oracle& o, size_t n,
     multiclass_logistic_regression_parameters params
     = multiclass_logistic_regression_parameters())
      : base(o), params(params),
        ds_ptr(new vector_dataset(o.datasource_info())),
        ds_o_ptr(NULL), ds(*ds_ptr), o(o),
        finite_seq(o.finite_list()), vector_seq(o.vector_list()),
        fixed_record(false),
        weights_(opt_variables::size_type
                 (label_->size(), o.finite_dim() - label_->size(),
                  label_->size(), o.vector_dim(), label_->size()),
                 0),
        obj_functor_ptr(NULL), grad_functor_ptr(NULL), prec_functor_ptr(NULL),
        gradient_method_ptr(NULL), gradient_ptr(NULL), iteration_(0),
        total_train_weight(0), nclasses_(label_->size()),
        train_acc(0), train_log_like(-std::numeric_limits<double>::max()),
        tmpvec(label_->size()),
        log_max_double(std::log(std::numeric_limits<double>::max()/nclasses_)) {
      if (params.method == 0 || params.method == 1) { // batch methods
        for (size_t i = 0; i < n; ++i)
          if (o.next())
            ds_ptr->insert(o.current());
          else
            break;
      }
      build();
    }

    ~multiclass_logistic_regression() {
      if (ds_ptr)
        delete(ds_ptr);
      if (ds_o_ptr)
        delete(ds_o_ptr);
      if (obj_functor_ptr)
        delete(obj_functor_ptr);
      if (grad_functor_ptr)
        delete(grad_functor_ptr);
      if (prec_functor_ptr)
        delete(prec_functor_ptr);
      if (gradient_method_ptr)
        delete(gradient_method_ptr);
      if (gradient_ptr)
        delete(gradient_ptr);
    }

    //! Train a new multiclass classifier of this type with the given data.
    boost::shared_ptr<multiclass_classifier> create(statistics& stats) const {
      boost::shared_ptr<multiclass_classifier>
        bptr(new multiclass_logistic_regression(stats, this->params));
      return bptr;
    }

    //! Train a new multiclass classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    boost::shared_ptr<multiclass_classifier> create(oracle& o, size_t n) const {
      boost::shared_ptr<multiclass_classifier>
        bptr(new multiclass_logistic_regression(o, n, this->params));
      return bptr;
    }

    // Getters and helpers
    //==========================================================================

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
      for (size_t i(0); i < finite_seq.size(); ++i) {
        if (sum(abs(weights_.f.column(i))) > params.convergence_zero)
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
      for (size_t i(0); i < vector_seq.size(); ++i) {
        if (sum(abs(weights_.v.column(i))) > params.convergence_zero)
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
    void fix_record(const record& r);

    /**
     * Undoes fix_record().
     */
    void unfix_record();

    /**
     * If available, this prints out statistics about the optimization done
     * during learning.
     */
    void print_optimization_stats(std::ostream& out) const {
      switch (params.method) {
      case 0:
      case 1:
      case 3:
        if (gradient_method_ptr)
          out << "multiclass_logistic_regression used a gradient_method:\n"
              << "\t iteration = " << gradient_method_ptr->iteration() << "\n"
              << "\t objective calls per iteration = "
              << gradient_method_ptr->objective_calls_per_iteration() << "\n";
        break;
      case 2:
        break;
      default:
        assert(false);
      }
    }

    // Learning and mutating operations
    //==========================================================================

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
    void add_gradient(opt_variables& grad, const record& r, double w) const;

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
    void add_expected_gradient(opt_variables& grad, const record& r,
                               const table_factor& fy, double w = 1.) const;

    /**
     * Used by add_combined_gradient from log_reg_crf_factor.
     * @todo Figure out a better way to do this.
     */
    void add_combined_gradient(opt_variables& grad, const record& r,
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
    add_expected_squared_gradient(opt_variables& hd, const record& r,
                                  const table_factor& fy, double w = 1.) const;

    // Prediction methods
    //==========================================================================

    //! Predict the label of a new example.
    std::size_t predict(const record& example) const {
      my_probabilities(example, tmpvec);
      return max_index(tmpvec, rng);
    }

    //! Predict the label of a new example.
    std::size_t predict(const assignment& example) const {
      my_probabilities(example, tmpvec);
      return max_index(tmpvec, rng);
    }

    //! Predict the probability of the class variable having each value.
    vec probabilities(const record& example) const {
      my_probabilities(example, tmpvec);
      return tmpvec;
    }

    //! Predict the probability of the class variable having each value.
    vec probabilities(const assignment& example) const {
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
    void reset_datasource(oracle& o, size_t n) { assert(false); }

    //! Resets the data source to be used in future rounds of training.
    //! ITERATIVE ONLY: This may be implemented by iterative learners.
    void reset_datasource(statistics& stats) { assert(false); }
    */

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
    (std::vector<vec>& lambdas, vec& means, vec& stderrs,
     const crossval_parameters<1>& cv_params, boost::shared_ptr<dataset> ds_ptr,
     const multiclass_logistic_regression_parameters& params,
     unsigned random_seed);

  }; // class multiclass_logistic_regression

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_LEARNING_DISCRIMINATIVE_MULTICLASS_LOGISTIC_REGRESSION_HPP
