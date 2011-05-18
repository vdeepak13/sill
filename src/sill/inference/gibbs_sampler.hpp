#ifndef SILL_GIBBS_SAMPLER_HPP
#define SILL_GIBBS_SAMPLER_HPP

#include <map>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>

#include <sill/factor/concepts.hpp>
#include <sill/math/permutations.hpp>
#include <sill/model/interfaces.hpp>

//#include <sill/range/transformed.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for doing Gibbs sampling for a factorized model.
   *
   * @tparam F  Type of factor.
   *
   * \author Joseph Bradley
   */
  template <typename F>
  class gibbs_sampler {

    // Public types
    //==========================================================================
  public:

    typedef F factor_type;
    typedef typename F::variable_type    variable_type;
    typedef typename F::domain_type      domain_type;
    typedef typename F::var_vector_type  var_vector_type;
    typedef typename F::assignment_type  assignment_type;

    /**
     * PARAMETERS
     *  - INIT_A (assignment_type): initial state of the sampler
     *     (default = each variable value chosen uniformly at random)
     *  - RANDOM_SEED (unsigned): used to make this algorithm deterministic
     *     (default = time)
     */
    class parameters {
    private:
      assignment_type init_a_;
      unsigned random_seed_;
    public:
      parameters() { }
      parameters& init_a(const assignment_type& value) {
        init_a_ = value; return *this;
      }
      parameters& random_seed(unsigned value) {
        random_seed_ = value; return *this;
      }
      const assignment_type& init_a() const { return init_a_; }
      unsigned random_seed() const { return random_seed_; }
    }; // class parameters

  protected:
    parameters params;

    //! random number generator
    boost::mt11213b rng;
    //! uniform distribution over [0, 1]
    boost::uniform_real<double> uniform_prob;

    //! The associated model
    const factorized_model<factor_type>& model;

    //! Factors in the associated model
    std::vector<factor_type> factors;
    //! Mapping from variables to factors over those variables
    //! var2factors[v] = indices of 'factors' which include variable v
    std::map<variable_type*, std::vector<size_t> > var2factors;

    //! Current assignment to variables
    assignment_type a;
    //! Last variable which was updated
    variable_type* last_v;

    //! Function returning the next variable to update
    virtual variable_type* next_variable() = 0;

    ///////////////////////// PUBLIC METHODS //////////////////////////////

  public:
    /**
     * Create a Gibbs sampler for the given model.
     */
    gibbs_sampler(const factorized_model<factor_type>& model,
                  const parameters& params = parameters())
      : params(params), model(model), a(params.init_a()), last_v(NULL) {

      rng.seed(static_cast<unsigned>(params.random_seed()));
      uniform_prob = boost::uniform_real<double>(0,1);

      factors.clear();
      foreach(const factor_type& f, model.factors())
        factors.push_back(f);

      size_t f_i = 0;
      foreach(const factor_type& f, factors) {
        foreach(variable_type* v, f.arguments()) {
          if (var2factors.count(v))
            var2factors[v].push_back(f_i);
          else
            var2factors[v] = std::vector<size_t>(1,f_i);
        }
        ++f_i;
      }
      // Assert that each variable has at least one associated factor.
      assert(var2factors.size() == model.arguments().size());

      // If no initial assignment was given, choose one uniformly at random.
      if (a.size() == 0) {
        foreach(variable_type* v, model.arguments()) {
          double r = uniform_prob(rng);
          for (size_t k = 1; k <= v->size(); ++k)
            if (r <= static_cast<double>(k) / (v->size())) {
              a[v] = k-1;
              break;
            }
        }
      }
    }

    virtual ~gibbs_sampler() { }

    /**
     * Get the next sample from the model (updating a single variable).
     */
    const assignment_type& next_sample() {
      last_v = next_variable();
      // Restrict the factors containing v, multiply them together,
      //  normalize the resulting factor, and then sample from it.
      assignment_type tmp_a(a);
      tmp_a.erase(last_v);
      std::vector<size_t> factor_indices(var2factors[last_v]);
      factor_type vfactor(factors[factor_indices[0]].restrict(tmp_a));
      for (size_t k = 1; k < factor_indices.size(); ++k)
        vfactor.combine_in(factors[factor_indices[k]].restrict(tmp_a),
                           product_op);
      vfactor.normalize();
      double r(uniform_prob(rng));
      size_t k = 0;
      foreach(double val, vfactor.values()) {
        if (r <= val) {
          a[last_v] = k;
          return a;
        } else {
          ++k;
          r -= val;
        }
      }
      assert(false);
      return a;
    }

    //! Return the current sample from the model.
    const assignment_type& current_sample() const {
      return a;
    }

    //! Return the last variable which was updated.
    //! This returns NULL if no samples have been taken.
    variable_type* last_variable() const { return last_v; }

    //! Reset the random seed.
    void random_seed(unsigned seed) {
      rng.seed(seed);
    }

  }; // class gibbs_sampler

  /**
   * Class for doing Gibbs sampling for a factorized model.
   * The variables are sampled in a fixed order.
   *
   * @tparam F  Type of factor.
   */
  template <typename F>
  class sequential_gibbs_sampler : public gibbs_sampler<F> {

    using gibbs_sampler<F>::rng;

    // Public types
    //==========================================================================
  public:

    typedef gibbs_sampler<F> base;

    typedef F factor_type;
    typedef typename F::variable_type    variable_type;
    typedef typename F::domain_type      domain_type;
    typedef typename F::var_vector_type  var_vector_type;
    typedef typename F::assignment_type  assignment_type;

    typedef typename base::parameters    parameters;

  public:
    /**
     * Create a Gibbs sampler for the given model.
     * @param var_order  order in which to sample the variables
     *                   (default = random)
     * @param params     parameters for the base class (gibbs_sampler)
     */
    sequential_gibbs_sampler(const factorized_model<factor_type>& model,
                             const var_vector_type& var_order
                             = var_vector_type(),
                             const parameters& params = parameters())
      : base(model, params), var_order(var_order), next_i(0) {
      var_vector_type args;
      foreach(variable_type* v, model.arguments())
        args.push_back(v);
      if (var_order.size() == 0) {
        std::vector<size_t> r(randperm(args.size(), rng));
        this->var_order.resize(args.size());
        for (size_t j = 0; j < r.size(); ++j)
          this->var_order[j] = args[r[j]];
      } else
        assert(var_order.size() == args.size());
    }

    // Protected data and methods
    //==========================================================================
  protected:

    //! Variables in order used for sampling
    var_vector_type var_order;

    //! Next index (into var_order) to return
    size_t next_i;

    //! Function returning the next variable to update
    variable_type* next_variable() {
      size_t this_i(next_i);
      ++next_i;
      if (next_i >= var_order.size())
        next_i = 0;
      return var_order[this_i];
    }

  }; // class sequential_gibbs_sampler

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
