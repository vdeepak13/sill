namespace sill {

  // concept
  template <typename Vec>
  struct OptimizationVector
    : DefaultConstructible<V>, CopyConstructible<V>, Assignable<V> { };

  template <typename Vec>
  struct opt_traits {
    typedef typename Vec::real_type real_type;
    typedef typename Vec::size_type size_type;

    static bool equal(const Vec& x, const Vec& y) {
      return x == y;
    }

    static size_type size(const Vec& x) {
      return x.size();
    }

    static Vec plus(const Vec& x, const Vec& y) {
      return x + y;
    }

    static Vec& plus_assign(const Vec& x, const Vec& y) {
      return x += y;
    }

    static value_type dot(const Vec& x, const Vec& y) {
      return dot(x, y);
    }

    // ...

  }; // struct opt_traits

  typedef boost::function<real_type(const Vec&)> objective_fn;
  typedef boost::function<const Vec&(const Vec&)> gradient_fn;
  typedef boost::function<void(const Vec&, Vec&)> preconditioner_fn;

  template <typename Vec>
  class gradient_descent {
    bool step() {
      gradient_fn(x, direction);
      direction *= -1.0;
      step_fn(x, direction);
    }
  };

  template <typename Vec>
  class conjugate_gradient {
    bool step() {
      if (iteration_ == 0) {
        gradient_fn(x, gradient);
        gradprec = gradient;
        preconditioner_fn(x, gradprec);
        direction = gradprec;
        step_fn(x, direction, gradient);
      } else {
        gradient_fn(x, gradient2);
        preconditioner_fn(x, gradprec2);
        real_type denom = dot(gradient, gradprec);
        gradient -= gradient2;
        real_type beta = dot(gradient, gradprec2) / (-denom);
        direction *= beta;
        direction -= gradprec;
        gradient.swap(gradient2);
        gradprec.swap(gradprec2);
        step_fn(x, direction, gradient);
      }
    }
  };

  template <typename Vec>
  class lbfgs {
    bool step() {

    }
  };

  template <typename Vec>
  class stochastic_gradient {
    
  };

  template <typename Vec>
  class opt_step {
    //! Applies a step to x in the given direction and returns the new
    //! objective value
    real_type apply(Vec& x, const Vec& direction, const Vec& gradient) = 0;

    //! Returns the number of calls to the objective function
    size_t calls_to_objective() const = 0;
  };

  template <typename Vec>
  class fixed_step {
    real_type apply(Vec& x, const Vec& direction, const Vec& gradient) {
      x += direction * eta;
    }
  };

  // implement the Wolfe conditions directly here
  template <typename Vec>
  class line_search {
    line_search(const objective_fn& objective,
                const param_type& params = param_type());
    real_type apply(Vec& x, const Vec& direction, const Vec& gradient) {
      // ...
      x += direction * eta;
    }

    real_type objective(real_type eta) const {
      //tmp = x + direction * eta;
      tmp = direction;
      tmp *= eta;
      tmp += x;
      return objective(tmp);
    }
  };

  template <typename Vec>
  class line_search_with_gradient : public line_search<Vec> {
    line_search(const objective_fn& objective,
                const gradient_fn& gradient,
                const param_type& params = param_type());

  };

  // crf parameter learning (eventually)
  class factor_mle<log_reg_crf_factor> {
  public:
    typedef multiclass_logistic_regression::param_type param_type;
    crf_parameter_learner(const dataset_type* dataset,
                          const param_type& params = param_type());
  };

  class multiclass_logistic_regression {
    struct param_type {
      gradient_method_factory::param_type gradient_params;
      double regularization;
    }
  };

} // namespace sill

