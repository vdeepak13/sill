#ifndef SILL_MARKOV_CHAIN_HPP
#define SILL_MARKOV_CHAIN_HPP

#include <sill/base/discrete_process.hpp>
#include <sill/learning/dataset/sequence_dataset.hpp>
#include <sill/math/logarithmic.hpp>

#include <sill/macros_def.hpp>

namespace sill {
  
  template <typename F>
  class markov_chain {
  public:
    typedef typename F::real_type           real_type;
    typedef logarithmic<real_type>          result_type;
    typedef typename F::variable_type       variable_type;
    typedef typename F::var_vector_type     var_vector_type;
    typedef discrete_process<variable_type> process_type;
    typedef std::set<process_type*>         domain_type;
    typedef std::vector<process_type*>      arg_vector_type;
    typedef typename F::assignment_type     assignment_type;
    typedef typename F::index_type          index_type;

    // LearnableFactor types
    typedef sequence_dataset<typename F::dataset_type> dataset_type;
    typedef typename dataset_type::record_type         record_type;

    //! Default constructor. Creates an empty chain
    markov_chain()
      : order_(0) { }
    
    //! Creates a chain with the given processes
    explicit markov_chain(const arg_vector_type& processes, size_t order = 1)
      : processes_(processes), order_(order) { }

    //! Creates a chain with the given initial distribution and transition model
    markov_chain(const F& initial, const F& transition)
      : processes_(make_vector(discrete_processes(initial.arguments()))) {
      this->initial(initial);
      this->transition(transition);
    }
    
    //! Returns the vector of processes associated with this chain
    const arg_vector_type& arg_vector() const {
      return processes_;
    }
    
    //! Returns the processes associated with this chain
    domain_type arguments() const {
      return make_domain(processes_);
    }

    //! Returns the variables representing the current state of this chain
    var_vector_type current() const {
      return variables(processes_, current_step);
    }
    
    //! Returns the vairables representing the next state of this chain
    var_vector_type next() const {
      return variables(processes_, next_step);
    }

    //! Returns the order of the chain
    size_t order() const {
      return order_;
    }
    
    //! Returns the initial distribution
    const F& initial() const {
      return initial_;
    }

    //! Return the transition model
    const F& transition() const {
      return transition_;
    }

    //! Sets the initial distribution
    void initial(const F& factor) {
      initial_ = factor.reorder(current());
    }

    //! Sets the transition model
    void transition(const F& factor) {
      transition_ = factor.reorder(concat(next(), current()));
    }

    //! Evaluates the probability of a record
    result_type operator()(const record_type& record) const {
      record.check_compatible(processes_);
      if (record.num_steps() == 0) { return result_type(1.0); }

      // evaluate the initial distribution
      factor_evaluator<F> eval_init(initial_);
      index_type index;
      record.extract(0, index);
      result_type likelihood = eval_init(index);
      if (record.num_steps() == 1) { return likelihood; }

      // evaluate the transition model
      factor_evaluator<F> eval_trans(transition_);
      for (size_t t = 1; t < record.num_steps(); ++t) {
        record.extract(t-1, t, index);
        likelihood *= eval_trans(index);
      }

      // account for the weight
      return pow(likelihood, record.weight());
    }
    
    //! Evaluates the log-likelihood of the model for the dataset
    //! \todo make this more efficient
    real_type log_likelihood(const dataset_type& ds) const {
      real_type ll = 0;
      foreach(const record_type& r, ds.records(processes_)) {
        ll += log(operator()(r));
      }
      return ll;
    }

    //! Draws a single random chain and stores it in a record
    template <typename RandomNumberGenerator>
    void sample(size_t len, record_type& record, RandomNumberGenerator& rng) const {
      record.check_compatible(processes_);

      // allocate the memory
      record.assign(len, 1.0);
      if (len == 0) { return; }

      // sample the initial state
      factor_sampler<F> init_sampler(initial_);
      index_type cur_state;
      init_sampler(cur_state, rng);
      record.set(0, cur_state);
      if (len == 1) { return; }

      // sample the transitions
      factor_sampler<F> trans_sampler(transition_, next());
      index_type next_state;
      for (size_t t = 1; t < record.num_steps(); ++t) {
        trans_sampler(next_state, cur_state, rng);
        record.set(t, next_state);
        cur_state.swap(next_state);
      }
    }

  private:
    arg_vector_type processes_;
    size_t order_;
    F initial_;
    F transition_;

  }; // class markov_chain

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
