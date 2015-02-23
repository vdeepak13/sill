#ifndef SILL_SOFTMAX_HPP
#define SILL_SOFTMAX_HPP

#include <sill/global.hpp>
#include <sill/base/assignment.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/base/variable_utils.hpp>
#include <sill/datastructure/finite_index.hpp>
#include <sill/datastructure/hybrid_index.hpp>
#include <sill/factor/base/factor.hpp>
#include <sill/factor/probability_array.hpp>
#include <sill/factor/traits.hpp>
#include <sill/factor/util/factor_mle.hpp>
#include <sill/learning/dataset/hybrid_dataset.hpp>
#include <sill/learning/dataset/hybrid_record.hpp>
#include <sill/math/constants.hpp>
#include <sill/math/function/softmax_param.hpp>
#include <sill/optimization/gradient_objective.hpp>
#include <sill/optimization/gradient_method/conjugate_gradient.hpp>
#include <sill/optimization/gradient_method/gradient_descent.hpp>
#include <sill/optimization/line_search/backtracking_line_search.hpp>
#include <sill/optimization/line_search/slope_binary_search.hpp>

#include <iostream>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A factor that represents a conditional distribution over a finite variable
   * given a set of vector variables. The conditional distribution is given by
   * a normalized exponential, p(y = j | x) \propto exp(b_j + x^T w_j).
   *
   * \tparam T a real type for representing each parameter
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename T = double>
  class softmax : public factor {
  public:
    // Public types
    //==========================================================================
    typedef T          real_type;
    typedef T          result_type;
    typedef variable   variable_type;
    typedef domain     domain_type;
    typedef var_vector var_vector_type;
    typedef assignment assignment_type;
    typedef softmax_param<T> param_type;

    /// IndexableFactor member types
    typedef hybrid_index<T> index_type;

    // LearnableFactor member types
    typedef hybrid_dataset<T> dataset_type;
    typedef hybrid_record<T> record_type;
    
    // Types to represent the parameters
    typedef typename softmax<T>::mat_type mat_type;
    typedef typename softmax<T>::vec_type vec_type;
 
    // Constructors and conversion operators
    //==========================================================================
    
    /**
     * Default constructor. Creates an empty factor.
     */
    softmax()
      : head_(NULL) { }

    /**
     * Constructs a factor with the given label variable and feature arguments.
     * Allocates the parameters but does not initialize their values.
     */
    softmax(finite_variable* head, const vector_var_vector& tail)
      : head_(NULL) {
      reset(head, tail);
    }

    /**
     * Constructs a factor with the given label variable and feature arguments.
     * Sets the parameters to to the given parameter vector.
     */
    softmax(finite_variable* head,
            const vector_var_vector& tail,
            const param_type& param)
      : head_(head),
        tail_(tail),
        param_(param) {
      assert(head);
      args_.insert(head);
      args_.insert(tail.begin(), tail.end());
      check_param();
    }

    /**
     * Exchanges the arguments and the parameters of two factors.
     */
    friend void swap(const softmax& f, const softmax& g) {
      if (&f != &g) {
        using std::swap;
        swap(f.args_, g.args_);
        swap(f.head_, g.head_);
        swap(f.tail_, g.tail_);
        swap(f.param_, g.param_);
      }
    }

    /**
     * Resets the content of this factor to the given finite and vector
     * arguments. The parameter values may become invalidated.
     */
    void reset(finite_variable* head, const finite_var_vector& tail) {
      assert(head);
      if (head_ != head || tail_ != tail) {
        args_.clear();
        args_.insert(head);
        args_.insert(tail.begin(), tail.end());
        head_ = head;
        tail_ = tail;
        param_.reset(head->size(), vector_size(tail));
      }
    }

    // Accessors and comparison operators
    //==========================================================================

    //! Returns the arguments set of this factor
    const domain& arguments() const {
      return args_;
    }

    //! Returns the label varaible or NUL if this factor is empty.
    finite_variable* head() const {
      return head_;
    }

    //! Returns the feature arguments of this factor.
    const vector_var_vector& tail() const {
      return tail_;
    }

    //! Returns true if the factor is empty.
    bool empty() const {
      return !head_;
    }

    //! Returns the number of arguments of this factor or 0 if the factor is empty.
    size_t arity() const {
      return args_.size();
    }

    //! Returns the number of assignments to the head variable.
    size_t num_labels() const {
      return param_.num_labels();
    }

    //! Returns the dimensionality of the underlying feature vector.
    size_t num_features() const {
      return param_.num_features();
    }

    //! Returns the parameters of this factor.
    const param_type& param() const {
      return param_;
    }

    //! Provides mutable access to the parameters of this factor.
    param_type& param() {
      return param_;
    }

    //! Returns the weight matrix.
    const mat_type& weight() const {
      return param_.weight();
    }

    //! Returns the bias vector.
    const vec_type& bias() const {
      return param_.bias();
    }

    /**
     * Returns the value of the factor for the given index.
     * The first finite value is assumed to be the label.
     */
    T operator()(const hybrid_index<T>& index) const {
      return param_(index.vector)[index.finite[0]];
    }

    /**
     * Returns the value of the factor (conditional probability) for the
     * given assignment.
     * \param strict if true, requires all the arguments to be present;
     *        otherwise, only the label variable must be present and the
     *        missing features are assumed to be 0.
     */
    T operator()(const assignment& a, bool strict = true) const {
      size_t finite = safe_get(a, head_);
      if (strict) {
        vec_type features;
        extract_features(a, features);
        return param_(features)[finite];
      } else {
        sparse_index<T> features;
        extract_features(a, features);
        return param_(features)[finite];
      }
    }

    /**
     * Returns the log-value of the factor for the given index.
     * The first finite value is assumed ot be the label.
     */
    T log(const hybrid_index<T>& index) const {
      return std::log(operator()(index));
    }

    /**
     * Returns the log of the value of the factor (conditional probability)
     * for the given assignment.
     * \param strict if true, requires all the arguments to be present;
     *        otherwise, only the label variable must be present and the
     *        missing features are assumed to be 0.
     */
    T log(const assignment& a, bool strict = true) const {
      return std::log(operator()(a, strict));
    }

    /**
     * Returns true if the two factors have the same argument vectors and
     * parameters.
     */
    friend bool operator==(const softmax& f, const softmax& g) {
      return f.head_ == g.head_ && f.tail_ == g.tail_ && f.param_ == g.param_;
    }

    /**
     * Returns true if the two factors do not have the same argument vectors
     * or parameters.
     */
    friend bool operator==(const softmax& f, const softmax& g) {
      return !(f == g);
    }

    // Indexing
    //==========================================================================

    /**
     * Extracts a dense feature vector from an assignment. All the tail
     * variables must be present in the assignment.
     */
    void extract_features(const vector_assignment& a, vec_type& result) const {
      result.resize(num_features());
      size_t row = 0;
      for (vector_variable* v : tail_) {
        vector_assignment::const_iterator it = a.find(v);
        if (it != a.end()) {
          std::copy(it->second.begin(), it->second.end(), &result[row]);
          row += v->size();
        } else {
          throw std::invalid_argument(
            "The assignment does not contain the variable " + v->str()
          );
        }
      }
    }
      
    /**
     * Extracts a sparse vector of features from an assignment. Tail variables
     * that are missing in the assignment are assumed to have a value of 0.
     */
    void extract_features(const vector_assignment& a,
                          sparse_index<T>& result) const {
      result.clear();
      result.reserve(vector_size(tail_));
      size_t id = 0;
      for (vector_variable* v : tail_) {
        vector_assignment::const_iterator it = a.find(v);
        if (it != a.end()) {
          for (size_t i = 0; i < v->size(); ++i) {
            result.emplace_back(id + i, it->second[i]);
          }
        }
        id += v->size();
      }
    }

    /**
     * Checks if the dimensions of the parameters match this factor's arguments.
     * \throw std::runtime_error if some of the dimensions do not match.
     */
    void check_param() const {
      if (empty()) {
        if (!param_.empty()) {
          throw std::runtime_error("The factor is empty but the parameters are not!");
        }
      } else {
        if (param_.num_labels() != head_->size()) {
          throw std::runtime_error("Invalid number of labels");
        }
        if (param_.num_features() != vector_size(tail_)) {
          throw std::runtime_error("Invalid number of features");
        }
      }
    }

    // Factor operations
    //==========================================================================

    /**
     * Returns true if the factor represents a valid distribution.
     * This is true if none of the parameters are infinite / nan.
     */
    bool is_normalizable() const {
      return param_.is_finite();
    }

    /**
     * Conditions the factor on the given features in the factor's internal
     * ordering of tail variables.
     */
    probability_array<T, 1>
    condition(const vec_type& index) const {
      return probability_array<T, 1>({head_}, param_(index));
    }

    /**
     * Conditions the factor on the assignment to its tail variables.
     * \param strict if true, requires that all the tail arguments are present
     *        in the assignment.
     */
    probability_array<T, 1>
    condition(const vector_assignment& a, bool strict = true) const {
      if (strict) {
        vec_type features;
        extract_features(a, features);
        return probability_array<T, 1>({head_}, param_(features));
      } else {
        sparse_index<T> features;
        extract_features(a, features);
        return probability_array<T, 1>({head_}, param_(features));
      }
    }

    /**
     * Returns the log-likelihood of the data under this model.
     */
    T log_likelihood(const hybrid_dataset<T>& ds) const {
      T result(0);
      foreach(const hybrid_record<T>& r, ds.records({head_}, tail_)) {
        result += r.weight * log(r.values);
      }
      return result;
    }

    /**
     * Returns the accuracy of predictions for this model.
     */
    T accuracy(const hybrid_dataset<T>& ds) const {
      T correct(0);
      T weight(0);
      foreach(const hybrid_record<T>& r, ds.records({head_}, tail_)) {
        arma::uword prediction;
        param_(r.values.vector).max(prediction);
        correct += r.weight * (prediction == r.values.finite[0]);
        weight += r.weight;
      }
      return correct / weight;
    }

    // Private members
    //==========================================================================
  private:
    //! The argument set of this factor
    domain args_;

    //! The head (label variable) or NULL if the factor is empty.
    finite_variable* head_;

    //! The tail (feature variables).
    vector_var_vector tail_;

    //! The underlying softmax function
    softmax<T> param_;
    
  }; // class softmax

  /**
   * Prints a human-readable representation of the CPD to a stream.
   * \relates softmax
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const softmax<T>& f) {
    out << "softmax(" << f.head() << "|" << f.tail() << ")" << std::endl
        << f.param();
    return out;
  }


  // Utility classes
  //==========================================================================
  /**
   * A utility class that represents a maximum-likelihood estimator of
   * a softmax conditional probability distribution. The maximum likelihood
   * estimate is computed iteratively using the specified optimization
   * class.
   */
  template <typename T>
  class factor_mle<softmax<T> > {
  public:
    // TODO: consider eliminating these
    typedef domain            domain_type;
    typedef var_vector        var_vector_type;
    typedef hybrid_dataset<T> dataset_type;
    typedef hybrid_record<T>  record_type;

    struct param_type {
      T regul;
      size_t max_iter;
      param_type(T regul = 0.1, size_t max_iter = 1000)
        : regul(regul), max_iter(max_iter) { }
    };
    
    /**
     * Creates a maximum-likelihood estimator with the given dataset,
     * parameters, and optimizer.
     */
    factor_mle(const hybrid_dataset<T>* ds,
               const param_type& params = param_type())
      : ds_(ds), params_(params) { }

    /**
     * Returns the conditional distribution p(head | tail) for a vector
     * tail, computed iteratively.
     */
    softmax<T> operator()(const finite_var_vector& head,
                              const vector_var_vector& tail) const {
      assert(head.size() == 1);
      //line_search<softmax<T> >* search = 
      //  new backtracking_line_search<softmax<T> >();
      line_search<softmax<T> >* search =
        new slope_binary_search<softmax<T>>(1e-6,
                                            wolfe_conditions<T>::param_type::conjugate_gradient());
      //typename gradient_descent<softmax<T> >::param_type gd_param;
      //gd_param.precondition = false;//true;
      //gradient_descent<softmax<T> > optimizer(search, gd_param);
      typename conjugate_gradient<softmax<T>>::param_type cg_param;
      cg_param.precondition = false;//true;
      conjugate_gradient<softmax<T> > optimizer(search, cg_param);
      softmax_objective objective(ds_, head, tail, params_.regul);
      optimizer.objective(&objective);
      optimizer.solution(softmax<T>(head[0]->size(), vector_size(tail), T(0)));
      size_t it = 0;
      while (!optimizer.converged() && it < params_.max_iter) {
        line_search_result<T> value = optimizer.iterate();
        std::cout << "Iteration " << it << ", " << value << std::endl;
          //                  << optimizer.solution();
        ++it;
      }
      if (!optimizer.converged()) {
        std::cerr << "Warning: failed to converge" << std::endl;
      }
      std::cout << "Number of calls: "
                << objective.value_calls << " "
                << objective.grad_calls << std::endl;
      return softmax<T>(head[0], tail, optimizer.solution());
    }

  private:
    //! Convergence and regularization parameters.
    param_type params_;

    //! The dataset.
    const hybrid_dataset<T>* ds_;
    
    /**
     * A class that iterates over the dataset, computing the value and
     * the derivatives of the softmax log-likelihood function.
     */
    class softmax_objective : public gradient_objective<softmax<T> > {
    public:
      softmax_objective(const hybrid_dataset<T>* ds,
                        const finite_var_vector& head,
                        const vector_var_vector& tail,
                        T regul)
        : ds_(ds),
          finite_(head),
          vector_(tail),
          g_(head[0]->size(), vector_size(tail)),
          h_(head[0]->size(), vector_size(tail)),
          regul_(regul),
          value_calls(0),
          grad_calls(0) { }

      T value(const softmax<T>& f) {
        T result(0);
        T weight(0);
        foreach (const hybrid_record<T>& r, ds_->records(finite_, vector_)) {
          result += r.weight * std::log(f(r.values.vector)[r.values.finite[0]]);
          weight += r.weight;
        }
        //std::cout << result << std::endl;
        result /= -weight;
        result += 0.5 * regul_ * dot(f, f);
        ++value_calls;
        return result;
      }

      const softmax<T>& gradient(const softmax<T>& f) {
        g_.zero();
        T weight(0);
        foreach (const hybrid_record<T>& r, ds_->records(finite_, vector_)) {
          g_.add_gradient(f, r.values.finite[0], r.values.vector, r.weight);
          weight += r.weight;
        }
        //std::cout << g_ << std::endl;
        g_ /= -weight;
        //g_ += regul * f;
        axpy(regul_, f, g_);
        ++grad_calls;
        return g_;
      }

      const softmax<T>& hessian_diag(const softmax<T>& f) {
        h_.zero();
        T weight(0);
        foreach (const hybrid_record<T>& r, ds_->records(finite_, vector_)) {
          h_.add_hessian_diag(f, r.values.vector, r.weight);
          weight += r.weight;
        }
        //std::cout << h_ << std::endl;
        h_ /= -weight;
        h_ += regul_;
        return h_;
      }

    private:
      const hybrid_dataset<T>* ds_;
      finite_var_vector finite_;
      vector_var_vector vector_;
      softmax<T> g_;
      softmax<T> h_;
      T regul_;

    public:
      size_t value_calls;
      size_t grad_calls;
    };

  }; // class factor_mle<softmax<T>>

} // namespace sill  

#include <sill/macros_undef.hpp>

#endif
