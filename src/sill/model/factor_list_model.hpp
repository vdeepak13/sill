#ifndef SILL_FACTOR_LIST_MODEL_HPP
#define SILL_FACTOR_LIST_MODEL_HPP

#include <sill/factor/concepts.hpp>
#include <sill/model/interfaces.hpp>

#include <sill/macros_def.hpp>

#include <sill/range/forward_range.hpp>


namespace sill {

  /**
   * Implements a factorized model which is simply a list of factors;
   * useful for a small representation which does not maintain a graph.
   * This is not necessarily a normalized distribution.
   *
   * \ingroup model
   */
  template <typename F>
  class factor_list_model
    : public factorized_model<F> {

    concept_assert((Factor<F>));

    // Public types
    //==========================================================================
  public:

    typedef factorized_model<F> base;

    typedef typename base::domain_type domain_type;
    
    typedef typename base::assignment_type assignment_type;

    // Public methods
    //==========================================================================
  public:

    //! Creates a factor list model with no factors.
    factor_list_model() { }

    //! Creates a factor list model with no factors but with the given
    //! variables.
    factor_list_model(const domain_type& args) : args(args) { }

    //! Creates a factor list model with the given factors.
    template <typename FactorRange>
    factor_list_model(const FactorRange& factors_)
      : factors_(boost::begin(factors_), boost::end(factors_)) {
      foreach(const F& f, factors_)
        args.insert(f.arguments());
    }

    //! Add a factor, adding new arguments as needed.
    void add_factor(const F& f) {
      factors_.push_back(f);
      args.insert(f.arguments().begin(), f.arguments().end());
    }

    //! Add factors, adding new arguments as needed.
    template <typename FactorRange>
    void add_factors(const FactorRange& new_factors) {
      foreach(const F& f, new_factors)
        add_factor(f);
    }

    //! Clear all factors and arguments.
    void clear() {
      factors_.clear();
      args.clear();
    }

    /**
     * Conditions this model on an assignment to one or
     * more of its variables. This is a mutable operation.
     * Note this does not normalize the distribution.
     *
     * @param assignment
     *        An assignment to some variables.  This assignment is
     *        instantiated in each factor.
     * \todo Should this combine factors which now have the same argument sets,
     *       or should there be another function for that?
     */
    factor_list_model& condition(const assignment_type& a) {

      // Compute the variables that are conditioned on.
      domain_type restricted_vars = a.keys().intersect(arguments());
      if (restricted_vars.empty())
        return *this;

      // For each factor which includes a variable which is being restricted,
      // (handling factors which no longer have any arguments via const_f)
      double const_f = 1;
      std::list<F> new_factors;
      foreach(F& f, factors_) {
        if (f.arguments().disjoint_from(restricted_vars))
          new_factors.push_back(f);
        else {
          if (f.arguments().subset_of(a.keys()))
            const_f *= f(a);
          else
            new_factors.push_back(f.restrict(a));
        }
      }
      if (new_factors.empty())
        new_factors.push_back(F(const_f));
      else
        new_factors.front() *= //.combine_in(
            constant_factor(const_f); //, product_op );
      factors_.swap(new_factors);

      args.remove(restricted_vars);

      return *this;
    }

    /////////////////////////////////////////////////////////////////
    // factorized_model<NodeF> interface
    /////////////////////////////////////////////////////////////////

    domain_type arguments() const { return args; }

    forward_range<const F&> factors() const { return factors_; }

    double log_likelihood(const assignment_type& a) const {
      using std::log;
      double result = 0;
      foreach(const F& factor, factors_) result += log(factor(a));
      return result;
    }

    logarithmic<double> operator()(const assignment_type& a) const {
      return logarithmic<double>(log_likelihood(a), log_tag());
    }

    //! Prints the arguments and factors of the model.
    template <typename OutputStream>
    void print(OutputStream& out) const {
      out << "Arguments: " << arguments() << "\n"
          << "Factors:\n";
      foreach(F f, factors()) out << f;
    }

    operator std::string() const {
      assert(false); // TODO
      // std::ostringstream out; out << *this; return out.str(); 
      return std::string();
    }

    // Private data
    //==========================================================================
  private:

    //! The factors in the model.
    std::list<F> factors_;

    //! The arguments to the model--note that some variables may not appear
    //! in factors.
    domain_type args;

  }; // factor_list_model

}

#include <sill/macros_undef.hpp>

#endif // SILL_FACTOR_LIST_MODEL_HPP
