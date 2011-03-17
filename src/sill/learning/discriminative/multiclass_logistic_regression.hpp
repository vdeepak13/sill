
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
#include <sill/math/linear_algebra.hpp>
#include <sill/math/statistics.hpp>
#include <sill/optimization/real_optimizer_builder.hpp>
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

    // Real optimization parameters
    //==========================================================================

    /**
     * 0 = batch gradient descent with line search,
     * 1 = batch conjugate gradient,
     * 2 = stochastic gradient descent (without line search)
     *  (default = 2)
     * 3 = batch conjugate gradient with a diagonal preconditioner
     */
//    size_t method;

    //! Learning rate in (0,1]
    //! This is only used to choose the initial step size
    //! if using a line search.
    //!  (default = .1)
//    double eta;

    /**
     * Rate at which to decrease the learning rate
     * (by multiplying ETA by MU each round).
     * This is not used if using a line search.
     *  (default = exp(log(10^-4) / INIT_ITERATIONS),
     *            or .999 if INIT_ITERATIONS == 0)
     */
//    double mu;

    //! Amount of change in average log likelihood
    //! below which algorithm will consider itself converged
    //!  (default = .000001)
//    double convergence_zero;

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

    /*
    double choose_mu() const {
      if (init_iterations == 0)
        return .999;
      else
        return exp(-4. * std::log(10.) / init_iterations);
    }
    */

    multiclass_logistic_regression_parameters()
      : regularization(2), lambda(.00001), init_iterations(1000),
        perturb(0), resolve_numerical_problems(false),
        random_seed(time(NULL)), debug(0),
        opt_method(real_optimizer_builder::CONJUGATE_GRADIENT),
        cg_update_method(0), lbfgs_M(10) { }
    //    method(2), eta(.1), mu(choose_mu()), convergence_zero(.000001),

    bool valid() const {
      if (regularization > 2)
        return false;
      if (lambda < 0)
        return false;
      if (perturb < 0)
        return false;
      if (!gm_params.valid())
        return false;
      if (cg_update_method > 0)
        return false;
      if (lbfgs_M == 0)
        return false;
      //      if (method > 3)        return false;
      //      if (eta <= 0 || eta > 1)        return false;
      //      if (mu <= 0 || mu > 1)        return false;
      //      if (convergence_zero < 0)        return false;
      return true;
    }

    void save(std::ofstream& out) const {
      out << regularization << " " << lambda << " " << init_iterations << " "
//        << method << " " << eta << " " << mu << " " << convergence_zero << " "
          << perturb << " " << (resolve_numerical_problems ? 1 : 0) << " "
          << random_seed << " " << debug << "\n";
    }

    void load(std::ifstream& in) {
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> regularization))
        assert(false);
      if (!(is >> lambda))
        assert(false);
      if (!(is >> init_iterations))
        assert(false);
      if (!(is >> perturb))
        assert(false);
      /*
      if (!(is >> method))
        assert(false);
      if (!(is >> eta))
        assert(false);
      if (!(is >> mu))
        assert(false);
      if (!(is >> convergence_zero))
        assert(false);
      */
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
      dense_matrix_type f;

      //! v(k,j) = weight for label k, vector index j
      dense_matrix_type v;

      //! Offsets b; b(k) = offset for label k
      dense_vector_type b;

      // Constructors
      //------------------------------------------------------------------------

      opt_variables() { }

      opt_variables(size_type s, double default_val = 0)
        : f(s.f_rows, s.f_cols, default_val),
          v(s.v_rows, s.v_cols, default_val),
          b(s.b_size, default_val) { }

      opt_variables(const dense_matrix_type& f, const dense_matrix_type& v,
                    const dense_vector_type& b)
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
                + sill::inner_prod(b, other.b));
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

    //! Objective functor usable with optimization routines.
    class objective_functor {

      const multiclass_logistic_regression& mlr;

      mutable typename dataset<la_type>::record_iterator ds_it;

      typename dataset<la_type>::record_iterator ds_end;

    public:
      objective_functor(const multiclass_logistic_regression& mlr)
        : mlr(mlr), ds_it(mlr.ds_ptr->begin()), ds_end(mlr.ds_ptr->end()) { }

      //! Computes the value of the objective at x.
      double objective(const opt_variables& x) const {
        double ll(0);
        size_t i(0);
        ds_it.reset();
        while (ds_it != ds_end) {
          dense_vector_type v;
          mlr.my_probabilities(*ds_it, v, x.f, x.v, x.b);
          const std::vector<size_t>& findata = (*ds_it).finite();
          size_t label_(findata[mlr.label_index_]);
          ll -= mlr.ds_ptr->weight(i) * std::log(v[label_]);
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

    //! Gradient functor used with optimization routines.
    class gradient_functor {

      const multiclass_logistic_regression& mlr;
      bool stochastic;

    public:
      gradient_functor(const multiclass_logistic_regression& mlr,
                       bool stochastic)
        : mlr(mlr), stochastic(stochastic) { }

      //! Computes the gradient of the function at x.
      //! @param grad  Data type in which to store the gradient.
      void gradient(opt_variables& grad, const opt_variables& x) const {
        if (stochastic)
          mlr.my_stochastic_gradient(grad, x);
        else
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
    typedef gradient_method<opt_variables,objective_functor,gradient_functor>
    gradient_method_type;

    typedef gradient_descent<opt_variables,objective_functor,gradient_functor>
    gradient_descent_type;

    typedef conjugate_gradient<opt_variables,objective_functor,gradient_functor>
    conjugate_gradient_type;

    typedef conjugate_gradient<opt_variables,objective_functor,
                               gradient_functor,preconditioner_functor>
    prec_conjugate_gradient_type;

    typedef lbfgs<opt_variables,objective_functor,gradient_functor>
    lbfgs_type;

    typedef stochastic_gradient<opt_variables,gradient_functor>
    stochastic_gradient_type;

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
//    const dataset<la_type>& ds;

    //! Oracle (for online/stochastic learning)
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
    gradient_functor* grad_functor_ptr;

    //! For all generic optimization methods
    preconditioner_functor* prec_functor_ptr;

    //! For batch gradient methods
    gradient_method_type* gradient_method_ptr;

    //! For stochastic optimization methods
    stochastic_gradient_type* stochastic_gradient_ptr;

    //! For stochastic gradient descent: gradients (to avoid reallocation)
//    opt_variables* gradient_ptr;

    //! For stochastic gradient descent: current eta (learning rate)
//    double eta;

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
    mutable dense_vector_type tmpvec;

    //! Stores std::log(std::numeric_limits<double>::max() / nclasses_).
    double log_max_double;

    // Protected methods
    //==========================================================================

    //! Initializes stuff using preset ds, o.
    void init(bool init_weights = true);

    //! Initialize optimization-related pointers (not ds,o pointers).
    void init_opt_pointers();

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
    void my_probabilities(const record_type& example, dense_vector_type& v) const;

    //! Set v to be the predicted class conditional probabilities for the
    //! given example.
    void my_probabilities(const assignment& example, dense_vector_type& v) const;

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
                     const record_type& example, double ex_weight, double alt_weight,
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

    /**
     * Compute a stochastic estimate of the gradient at x, storing it in
     * the given opt_variables.
     * Computes some objectives for little extra cost.
     * @param gradient  Pre-allocated place to store gradient.
     * @return <avg training accuracy, avg training log likelihood>
     *         (average over stochastically chosen samples)
     */
    std::pair<double,double>
    my_stochastic_gradient(opt_variables& gradient,
                           const opt_variables& x) const;

    //! Compute the diagonal of the Hessian at x, storing it in the given
    //! opt_variables.
    //! @param hd    Pre-allocated place to store the Hessian diagonal.
    void
    my_hessian_diag(opt_variables& hd, const opt_variables& x) const;

    //! Take one step of a batch gradient method.
    //! @return  true iff learner may be trained further (and has not converged)
//    bool step_gradient_method();

    //! Take one step of stochastic gradient descent.
    //! @todo How should we estimate convergence?
    //!       (Estimate change based on a certain number of previous examples?)
    //! @return  true iff learner may be trained further (and has not converged)
//    bool step_stochastic_gradient_descent();

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
      init();
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
        my_ds_ptr(NULL),
        my_ds_o_ptr(new ds_oracle<la_type>(stats.get_dataset())),
        ds_ptr(&(stats.get_dataset())), o_ptr(my_ds_o_ptr) {
      init();
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
        my_ds_ptr(new vector_dataset<la_type>(o.datasource_info())),
        my_ds_o_ptr(NULL), ds_ptr(my_ds_ptr), o_ptr(&o) {
      init();
      if (!real_optimizer_builder::is_stochastic(params.opt_method)) {
        for (size_t i = 0; i < n; ++i)
          if (o_ptr->next())
            my_ds_ptr->insert(o_ptr->current());
          else
            break;
      }
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
        my_ds_ptr(NULL), my_ds_o_ptr(new ds_oracle<la_type>(ds)),
        ds_ptr(&ds), o_ptr(my_ds_o_ptr) {
      init();
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
        my_ds_ptr(NULL), my_ds_o_ptr(new ds_oracle<la_type>(ds)),
        ds_ptr(&ds), o_ptr(my_ds_o_ptr),
        weights_(init_mlr.weights_) {
      init(false);
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
    using base::test_log_likelihood;

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
        if (sum(abs(weights_.f.column(i))) > params.gm_params.convergence_zero)
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
        if (sum(abs(weights_.v.column(i))) > params.gm_params.convergence_zero)
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
        if (gradient_method_ptr)
          out << "multiclass_logistic_regression used a gradient_method:\n"
              << "\t iteration = " << gradient_method_ptr->iteration() << "\n"
              << "\t objective calls per iteration = "
              << gradient_method_ptr->objective_calls_per_iteration() << "\n";
        break;
      case real_optimizer_builder::STOCHASTIC_GRADIENT:
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
    void add_gradient(opt_variables& grad, const record_type& r, double w) const;

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
    typedef vector<value_type>           dense_vector_type;

    // Constructors and destructors
    // =========================================================================

    mlr_validation_functor()
      : mlr_ptr(NULL), do_cv(false) { }

    //! Constructor for testing with the given parameters.
    explicit mlr_validation_functor
    (const multiclass_logistic_regression_parameters& params)
      : params(params), mlr_ptr(NULL), do_cv(false) { }

    //! Constructor for testing via CV, where each round chooses lambda via
    //! a second level of CV.
    mlr_validation_functor
    (const multiclass_logistic_regression_parameters& params,
     const crossval_parameters& cv_params)
      : params(params), mlr_ptr(NULL), do_cv(true), cv_params(cv_params) { }

    /*
    //! Constructor which warm starts immediately.
    mlr_validation_functor
    (const multiclass_logistic_regression_parameters& params,
     const multiclass_logistic_regression& mlr)
      : params(params), mlr_ptr(new multiclass_logistic_regression(mlr)) {
      use_weights = true;
    }
    */

    ~mlr_validation_functor() {
      if (mlr_ptr)
        delete(mlr_ptr);
    }

    mlr_validation_functor(const mlr_validation_functor& other)
      : params(other.params), mlr_ptr(NULL), do_cv(other.do_cv),
        cv_params(other.cv_params) {
      if (other.mlr_ptr)
        mlr_ptr = new multiclass_logistic_regression<la_type>(*(other.mlr_ptr));
    }

    mlr_validation_functor& operator=(const mlr_validation_functor& other) {
      params = other.params;
      mlr_ptr = NULL;
      if (other.mlr_ptr)
        mlr_ptr = new multiclass_logistic_regression<la_type>(*(other.mlr_ptr));
      do_cv = other.do_cv;
      cv_params = other.cv_params;
      return *this;
    }

    //! Like a constructor.
    void reset(const multiclass_logistic_regression_parameters& params) {
      this->params = params;
      if (mlr_ptr)
        delete(mlr_ptr);
      mlr_ptr = NULL;
      do_cv = false;
    }

    //! Like a constructor.
    void reset(const multiclass_logistic_regression_parameters& params,
               const crossval_parameters& cv_params) {
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
        params.lambda =
          multiclass_logistic_regression<la_type>::choose_lambda
          (cv_params, ds, params, unif_int(rng));
        random_seed = unif_int(rng);
      }
      params.random_seed = random_seed;
      if (mlr_ptr) {
        mlr_ptr =
          new multiclass_logistic_regression<la_type>(ds, *mlr_ptr, params);
      } else {
        mlr_ptr = new multiclass_logistic_regression<la_type>(ds, params);
      }
    }

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
      value_type ll = mlr_ptr->test_log_likelihood(ds).first;
      result_map_[prefix + "log likelihood"] = ll;
      result_map_[prefix + "accuracy"] = mlr_ptr->test_accuracy(ds);
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
  void multiclass_logistic_regression<LA>::init(bool init_weights) {
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
    gradient_method_ptr = NULL;
    //      gradient_ptr = NULL;
    iteration_ = 0;
    total_train_weight = 0;
    train_acc = 0;
    train_log_like = -std::numeric_limits<double>::max();
    if (label_) {
      nclasses_ = label_->size();
      tmpvec.resize(label_->size());
    }
    if (nclasses_ != 0)
      log_max_double = std::log(std::numeric_limits<double>::max()/nclasses_);
    else
      log_max_double = std::log(std::numeric_limits<double>::max());
  } // init

  template <typename LA>
  void multiclass_logistic_regression<LA>::init_opt_pointers() {
    switch(params.opt_method) {
    case real_optimizer_builder::CONJUGATE_GRADIENT_DIAG_PREC:
      prec_functor_ptr = new preconditioner_functor(*this);
    case real_optimizer_builder::GRADIENT_DESCENT:
    case real_optimizer_builder::CONJUGATE_GRADIENT:
    case real_optimizer_builder::LBFGS:
      obj_functor_ptr = new objective_functor(*this);
      grad_functor_ptr = new gradient_functor(*this, false);
      break;
    case real_optimizer_builder::STOCHASTIC_GRADIENT:
      grad_functor_ptr = new gradient_functor(*this, true);
      break;
    default:
      assert(false);
    }
    switch (params.opt_method) {
    case real_optimizer_builder::GRADIENT_DESCENT:
      {
        gradient_descent_parameters ga_params;
        gradient_method_ptr =
          new gradient_descent_type(*obj_functor_ptr, *grad_functor_ptr,
                                    weights_, ga_params);
      }
      break;
    case real_optimizer_builder::CONJUGATE_GRADIENT:
      {
        conjugate_gradient_parameters cg_params;
        gradient_method_ptr =
          new conjugate_gradient_type(*obj_functor_ptr, *grad_functor_ptr,
                                      weights_, cg_params);
      }
      break;
    case real_optimizer_builder::CONJUGATE_GRADIENT_DIAG_PREC:
      {
        conjugate_gradient_parameters cg_params;
        gradient_method_ptr =
          new prec_conjugate_gradient_type(*obj_functor_ptr, *grad_functor_ptr,
					   *prec_functor_ptr, weights_,
					   cg_params);
      }
      break;
    case real_optimizer_builder::LBFGS:
      {
        lbfgs_parameters lbfgs_params;
        gradient_method_ptr =
          new lbfgs_type(*obj_functor_ptr, *grad_functor_ptr,
                         weights_, lbfgs_params);
      }
      break;
    case real_optimizer_builder::STOCHASTIC_GRADIENT:
      {
        stochastic_gradient_parameters sg_params;
        if (params.init_iterations != 0)
          sg_params.step_multiplier =
            std::exp(std::log(.0001) / params.init_iterations);
        stochastic_gradient_ptr =
          new stochastic_gradient_type(*grad_functor_ptr, weights_, sg_params);
      }
      break;
    default:
      assert(false);
    }
  } // init_opt_pointers

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
    if (gradient_method_ptr)
      delete(gradient_method_ptr);
    gradient_method_ptr = NULL;
    if (stochastic_gradient_ptr)
      delete(stochastic_gradient_ptr);
    stochastic_gradient_ptr = NULL;
    //      if (gradient_ptr)    delete(gradient_ptr);
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
    rng.seed(static_cast<unsigned>(params.random_seed));
    for (size_t i(0); i < ds.size(); ++i)
      total_train_weight += ds.weight(i);

    // Initialize weights
    if (params.perturb > 0) {
      boost::uniform_real<double> uniform_dist;
      uniform_dist = boost::uniform_real<double>
        (-1 * params.perturb, params.perturb);
      for (size_t i = 0; i < weights_.f.size1(); ++i)
        for (size_t j = 0; j < weights_.f.size2(); ++j)
          weights_.f(i,j) = uniform_dist(rng);
      for (size_t i = 0; i < weights_.v.size1(); ++i)
        for (size_t j = 0; j < weights_.v.size2(); ++j)
          weights_.v(i,j) = uniform_dist(rng);
      foreach(double& v, weights_.b)
        v = uniform_dist(rng);
    }

    init_opt_pointers();

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
      for (size_t k(0); k < nclasses_; ++k) {
        if (v(k) > log_max_double) {
          double maxval(v(max_index(v, rng)));
          for (size_t k(0); k < nclasses_; ++k) {
            if (v(k) == maxval)
              v(k) = 1;
            else
              v(k) = 0;
          }
          v /= sum(v);
          return;
        }
      }
      for (size_t k(0); k < nclasses_; ++k)
        v(k) = exp(v(k));
      double tmpsum(sum(v));
      if (tmpsum == 0) {
        v.ones();
        v /= nclasses_;
        return;
      }
      v /= tmpsum;
    } else {
      for (size_t k(0); k < nclasses_; ++k) {
        if (v(k) > log_max_double) {
          throw std::runtime_error
            (std::string("multiclass_logistic_regression") +
             " had overflow when computing probabilities.  To deal with such" +
             " overflows in a hacky (but reasonable) way, use the parameter" +
             " resolve_numerical_problems.");
        }
        v(k) = exp(v(k));
      }
      if (sum(v) == 0)
        throw std::runtime_error
          (std::string("multiclass_logistic_regression") +
           " got all zeros when computing probabilities.  To deal with such" +
           " issues in a hacky (but reasonable) way, use the parameter" +
           " resolve_numerical_problems.");
      v /= sum(v);
    }
  }

  template <typename LA>
  void
  multiclass_logistic_regression<LA>::
  my_probabilities(const record_type& example, dense_vector_type& v,
                   const dense_matrix_type& w_fin_,
                   const dense_matrix_type& w_vec_,
                   const dense_vector_type& b_) const {
    v = b_;
    const std::vector<size_t>& findata = example.finite();
    for (size_t k(0); k < nclasses_; ++k) {
      for (size_t j(0); j < finite_indices.size(); ++j) {
        size_t val(findata[finite_indices[j]]);
        v(k) += w_fin_(k, finite_offset[j] + val);
      }
    }
    if (w_vec_.size() != 0)
      v += w_vec_ * example.vector();
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
    for (size_t k(0); k < nclasses_; ++k) {
      for (size_t j(0); j < finite_indices.size(); ++j) {
        size_t val(safe_get(fa, finite_seq[finite_indices[j]]));
        v(k) += w_fin_(k, finite_offset[j] + val);
      }
      for (size_t j(0); j < vector_seq.size(); ++j) {
        const vec& vecdata = safe_get(va, vector_seq[j]);
        for (size_t j2(0); j2 < vector_seq[j]->size(); ++j2) {
          size_t ind(vector_offset[j] + j2);
          v(k) += w_vec_(k,ind) * vecdata[j2];
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
                   const opt_variables& x) const {
    dense_vector_type v;
    my_probabilities(example, v, x.f, x.v, x.b);
    const std::vector<size_t>& findata = example.finite();
    const vector_type& vecdata = example.vector();
    size_t label_val(findata[label_index_]);
    size_t pred_(max_index(v, rng));
    if (label_val == pred_)
      acc += weight;
    ll += weight * std::log(v[label_val]);
    // Update gradients
    v(label_val) -= 1;
    v *= weight * alt_weight;
    for (size_t j(0); j < finite_indices.size(); ++j) {
      size_t val(finite_offset[j] + findata[finite_indices[j]]);
      for (size_t k(0); k < nclasses_; ++k) {
        gradient.f(k, val) += v(k);
      }
    }
    if (vecdata.size() != 0)
      gradient.v += outer_product(v, vecdata);
    gradient.b += v;
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
      for (size_t i(0); i < nclasses_; ++i) {
        for (size_t j(0); j < x.f.size2(); ++j) {
          if (x.f(i,j) > 0)
            gradient.f(i,j) += w;
          else if (x.f(i,j) < 0)
            gradient.f(i,j) -= w;
        }
        for (size_t j(0); j < x.v.size2(); ++j) {
          if (x.v(i,j) > 0)
            gradient.v(i,j) += w;
          else if (x.v(i,j) < 0)
            gradient.v(i,j) -= w;
        }
        if (x.b(i) > 0)
          gradient.b(i) += w;
        else if (x.b(i) < 0)
          gradient.b(i) -= w;
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
  std::pair<double,double>
  multiclass_logistic_regression<LA>::my_gradient(opt_variables& gradient,
                                              const opt_variables& x) const {
    double train_acc(0.);
    double train_log_like(0.);
    gradient.zeros();
    typename dataset<la_type>::record_iterator it_end(ds_ptr->end());
    size_t i(0); // index into dataset
    for (typename dataset<la_type>::record_iterator it(ds_ptr->begin());
         it != it_end; ++it) {
      // Compute v = prediction for *it.  Update accuracy, log likelihood.
      add_raw_gradient
        (gradient, train_acc, train_log_like, *it, ds_ptr->weight(i), 1, x);
      ++i;
    }
    gradient.f /= total_train_weight;
    gradient.v /= total_train_weight;
    gradient.b /= total_train_weight;

    // Update gradients to account for regularization
    add_reg_gradient(gradient, 1, x);
    return std::make_pair(train_acc, train_log_like);
  } // end of function my_gradient()

  template <typename LA>
  std::pair<double,double>
  multiclass_logistic_regression<LA>::
  my_stochastic_gradient(opt_variables& gradient,
                         const opt_variables& x) const {
    double train_acc(0.);
    double train_log_like(0.);
    gradient.zeros();
    if (o_ptr->next()) {
      const record_type& r = o_ptr->current();
      double r_weight(o_ptr->weight());
//      total_train_weight += r_weight;
      add_raw_gradient(gradient, train_acc, train_log_like, r, r_weight, 1, x);
      // Update gradients to account for regularization
      add_reg_gradient(gradient, 1, x);
      /*
        if (params.debug > 1) {
        std::cerr << "Gradient extrema info:\n";
        gradient.print_extrema_info(std::cerr);
        }
      */
    }
    return std::make_pair(train_acc, train_log_like);
  }

  template <typename LA>
  void
  multiclass_logistic_regression<LA>::my_hessian_diag
  (opt_variables& hd, const opt_variables& x) const {

    if (hd.size() != x.size())
      hd.resize(x.size());
    hd.zeros();
    typename dataset<la_type>::record_iterator it_end(ds_ptr->end());
    size_t i(0); // index into ds
    dense_vector_type v;
    vector_type vecdata;
    for (typename dataset<la_type>::record_iterator it(ds_ptr->begin());
         it != it_end; ++it) {
      my_probabilities(*it, v, x.f, x.v, x.b);
      v -= elem_mult(v, v);
      v *= ds_ptr->weight(i);
      const std::vector<size_t>& findata = (*it).finite();
      for (size_t j(0); j < finite_indices.size(); ++j) {
	size_t val(findata[finite_indices[j]]);
	hd.f.add_column(finite_offset[j] + val, v);
      }
      elem_mult_out((*it).vector(), (*it).vector(), vecdata); // COULD BE FASTER FOR SPARSE DATA
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

  /*
  template <typename LA>
  bool multiclass_logistic_regression<LA>::step_stochastic_gradient_descent() {
    if (!gradient_ptr)
      return false;
    if (!(o_ptr->next()))
      return false;
    const record_type& r = o_ptr->current();
    double r_weight(o_ptr->weight());
    total_train_weight += r_weight;
    // Compute v = prediction for *it.  Update accuracy, log likelihood.
    gradient_ptr->f.zeros_memset();
    gradient_ptr->v.zeros_memset();
    gradient_ptr->b.zeros_memset();
    add_raw_gradient(*gradient_ptr, train_acc, train_log_like, r, r_weight,
                     eta, weights_);
    // Update gradients to account for regularization
    add_reg_gradient(*gradient_ptr, eta, weights_);

    if (params.debug > 1) {
      std::cerr << "Gradient extrema info:\n";
      gradient_ptr->print_extrema_info(std::cerr);
    }

    // Update weights and learning rate eta.
    //  (Note that the gradient has already been multiplied by eta.)
    weights_.f -= gradient_ptr->f;
    weights_.v -= gradient_ptr->v;
    weights_.b -= gradient_ptr->b;
    eta *= params.mu;

    if (params.debug > 1) {
      std::cerr << "Weights extrema info:\n";
      weights_.print_extrema_info(std::cerr);
    }

    ++iteration_;
    return true;
  } // end of function: void step_stochastic_gradient_descent()
  */

  // Constructors, etc.
  //==========================================================================

  template <typename LA>
  multiclass_logistic_regression<LA>&
  multiclass_logistic_regression<LA>::
  operator=(const multiclass_logistic_regression& other) {
    this->clear_pointers();

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

    // To do eventually: Do a more careful deep copy of gradient_method_ptr,
    //  stochastic_gradient_ptr, but have them reference the *_functor_ptr
    //  copies.
    init_opt_pointers();

    iteration_ = other.iteration_;
    total_train_weight = other.total_train_weight;
    nclasses_ = other.nclasses_;
    train_acc = other.train_acc;
    train_log_like = other.train_log_like;
    tmpvec = other.tmpvec;
    log_max_double = other.log_max_double;

    return *this;
  }

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
    for (size_t j(0); j < finite_indices.size(); ++j) {
      size_t val(safe_get(fa, finite_seq[finite_indices[j]]));
      grad.f(label_val, finite_offset[j] + val) -= w;
    }
    for (size_t j(0); j < vector_seq.size(); ++j) {
      const vector_type& vecdata = safe_get(va, vector_seq[j]);
      for (size_t j2(0); j2 < vecdata.size(); ++j2) {
        size_t ind(vector_offset[j] + j2);
        grad.v(label_val, ind) -= vecdata[j2] * w;
      }
    }
    grad.b(label_val) -= w;
  }

  template <typename LA>
  void multiclass_logistic_regression<LA>::add_gradient(opt_variables& grad,
                                                    const record_type& r,
                                                    double w) const {
    if ((finite_offset.size() != 0 &&
         r.finite_numbering_ptr->size() != finite_offset.size()) ||
        (vector_offset.size() != 0 &&
         r.vector_numbering_ptr->size() != vector_offset.size())) {
      add_gradient(grad, r.assignment(), w);
    } else {
      const std::vector<size_t>& findata = r.finite();
      size_t label_val(findata[label_index_]);
      for (size_t j(0); j < finite_indices.size(); ++j) {
        size_t val(findata[finite_indices[j]]);
        grad.f(label_val, finite_offset[j] + val) -= w;
      }
      grad.v.subtract_row(label_val, r.vector() * w);
      grad.b(label_val) -= w;
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
    dense_vector_type r_vector(grad.v.size2());
    vector_assignment2vector(a.vector(), vector_seq, r_vector);
    for (size_t label_val(0); label_val < nclasses_; ++label_val) {
      tmpa[label_] = label_val;
      double label_prob(label_marginal(tmpa));
      if (label_prob == 0)
        continue;
      label_prob *= w;
      grad.v.subtract_row(label_val, label_prob * r_vector);
      grad.b(label_val) -= label_prob;
    }
    tmpa = a.finite();
    foreach(const finite_assignment& fa, assignments(fy.arguments())) {
      finite_assignment::const_iterator fa_end(fa.end());
      for (finite_assignment::const_iterator fa_it(fa.begin());
           fa_it != fa_end; ++fa_it)
        tmpa[fa_it->first] = fa_it->second;
      size_t label_val(tmpa[label_]);
      for (size_t j(0); j < finite_indices.size(); ++j) {
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
      add_expected_gradient(grad, r.assignment(), fy, w);
    } else {
      // Get marginal over label variable.
      table_factor
        label_marginal(fy.marginal(make_domain<finite_variable>(label_)));
      finite_assignment tmpa;
      for (size_t label_val(0); label_val < nclasses_; ++label_val) {
        tmpa[label_] = label_val;
        double label_prob(label_marginal(tmpa));
        if (label_prob == 0)
          continue;
        label_prob *= w;
        grad.v.subtract_row(label_val, label_prob * r.vector());
        grad.b(label_val) -= label_prob;
      }
      tmpa = r.finite_assignment();
      foreach(const finite_assignment& fa, assignments(fy.arguments())) {
        finite_assignment::const_iterator fa_end(fa.end());
        for (finite_assignment::const_iterator fa_it(fa.begin());
             fa_it != fa_end; ++fa_it)
          tmpa[fa_it->first] = fa_it->second;
        size_t label_val(tmpa[label_]);
        for (size_t j(0); j < finite_indices.size(); ++j) {
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
      assignment tmpa(r.assignment());
      add_gradient(grad, tmpa, w);
      add_expected_gradient(grad, tmpa, fy, -w);
    } else {
      // Get marginal over label variable.
      table_factor
        label_marginal(fy.marginal(make_domain<finite_variable>(label_)));
      finite_assignment tmpa;
      const std::vector<size_t>& findata = r.finite();
      for (size_t label_val(0); label_val < nclasses_; ++label_val) {
        tmpa[label_] = label_val;
        double label_prob(label_marginal(tmpa));
        if (label_prob == 0)
          continue;
        if (label_val == findata[label_index_])
          label_prob -= 1.;
        label_prob *= -w;
        grad.v.subtract_row(label_val, label_prob * r.vector());
        grad.b(label_val) -= label_prob;
      }
      for (size_t j(0); j < finite_indices.size(); ++j) {
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
        for (size_t j(0); j < finite_indices.size(); ++j) {
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
      for (size_t label_val(0); label_val < nclasses_; ++label_val) {
        tmpa[label_] = label_val;
        double label_prob(label_marginal(tmpa));
        if (label_prob == 0)
          continue;
        label_prob *= w;
        hd.v.subtract_row(label_val,
                          label_prob * elem_mult(r.vector(),r.vector()));
        hd.b(label_val) -= label_prob;
      }
      tmpa = r.finite_assignment();
      foreach(const finite_assignment& fa, assignments(fy.arguments())) {
        finite_assignment::const_iterator fa_end(fa.end());
        for (finite_assignment::const_iterator fa_it(fa.begin());
             fa_it != fa_end; ++fa_it)
          tmpa[fa_it->first] = fa_it->second;
        size_t label_val(tmpa[label_]);
        for (size_t j(0); j < finite_indices.size(); ++j) {
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
    if (gradient_method_ptr) {
      double prev_train_log_like(train_log_like);
      if (!gradient_method_ptr->step())
        return false;
      train_log_like = - gradient_method_ptr->objective();
      if (train_log_like < prev_train_log_like) {
        if (params.debug > 0)
          std::cerr << "multiclass_logistic_regression took a step which "
                    << "lowered the regularized log likelihood from "
                    << prev_train_log_like << " to " << train_log_like
                    << std::endl;
      }
      if (params.debug > 1) {
        std::cerr << "change in regularized log likelihood = "
                  << (train_log_like - prev_train_log_like) << std::endl;
      }
      // Check for convergence
      if (fabs(train_log_like - prev_train_log_like)
          < params.gm_params.convergence_zero) {
        if (params.debug > 1)
          std::cerr << "multiclass_logistic_regression converged:"
                    << " regularized training log likelihood changed from "
                    << prev_train_log_like << " to " << train_log_like
                    << "; exiting early (iteration " << iteration() << ")."
                    << std::endl;
        return false;
      }
    } else if (stochastic_gradient_ptr) {
      if (!stochastic_gradient_ptr->step())
        return false;
    } else {
      assert(false);
    }
    ++iteration_;
    return true;
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
    out << train_acc << " " << train_log_like
        << " " << iteration_ << " " << total_train_weight << "\n";
    for (size_t i = 0; i < weights_.f.size1(); ++i)
      out << weights_.f.row(i) << " ";
    out << "\n";
    for (size_t i = 0; i < weights_.f.size1(); ++i)
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
    if (!(is >> train_log_like))
      assert(false);
    if (!(is >> iteration_))
      assert(false);
    if (!(is >> total_train_weight))
      assert(false);
    getline(in, line);
    is.clear();
    is.str(line);
    weights_.f.resize(nclasses_, ds.finite_dim() - nclasses_);
    weights_.v.resize(nclasses_, ds.vector_dim());
    for (size_t j = 0; j < nclasses_; ++j) {
      read_vec(is, tmpvec);
      weights_.f.set_row(j, tmpvec);
    }
    getline(in, line);
    is.clear();
    is.str(line);
    for (size_t j = 0; j < nclasses_; ++j) {
      read_vec(is, tmpvec);
      weights_.v.set_row(j, tmpvec);
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

} // end of namespace: prl

#include <sill/macros_undef.hpp>

#endif // #ifndef _SILL_MULTICLASS_LOGISTIC_REGRESSION_HPP_
