namespace sill {

  class finite_timed_dataset {

    typedef finite_discrete_process    argument_type;
    typedef finite_timed_domain     domain_type;
    typedef finite_timed_vector     vector_type;
    typedef finite_timed_assignment assignment_type;
    typedef finite_timed_record     record_type;

    typedef timed_record_iterator<finited_timed_dataset>
      record_iterator;
    typedef timed_const_record_iterator<finite_timed_dataset>
      const_record_iterator;
    
    typedef finite_dataset static_dataset;

    finite_timed_dataset() { }

    void initialize(const finite_timed_vector& args);

    size_t size() const;
    
    finite_timed_domain arguments() const;

    std::pair<record_iterator, record_iterator>
    records(const finite_timed_vector& args);

    std::pair<const_record_iterator, const_record_iterator>
    records(const finite_timed_vector& args) const;

    sliding_view<finite_timed_dataset>
    sliding(finite_discrete_process p, size_t history) const;

    fixed_view<finite_timed_dataset>
    fixed(size_t time = 0) const;
  };

  template <typename DS>
  class finite_sliding_view : public typename DS::static_dataset { };

  template <typename DS>
  class finite_fixed_view : public typename DS::static_dataset { };

  // eventually make a variadic template
  template <typename DS1, typename DS2>
  class hybrid_dataset
    : public DS1, public DS2 {
    
  };

  /*
    Example: 
    parameter learning of dynamic Bayesian networks (fully observed case):
    - like learning a BN, but tied parameters
    - sliding view over a subset of variables (child and its parents)
    - when the DBN factor is moment_gaussian, table_factor, hybrid<moment_gaussian>
      etc., can do this efficiently using sliding view of hybrid dataset
    
    parameter learning of HMMs (fully observed case):
    - same as learning DBNs
    - sliding view over two successive time steps to learn the transition model
    - sliding view over individual time steps to learn the observation model
    
    Baum-Welch:
    - for each observation sequence, compute the conditional probability of the
      hidden states, given the observation; this gives us the virtual counts
    - recompute the prior model (normalized sum of the conditional joints)
    - recompute the observation model (sum of observations weighted by the
      posterior probability of the state given the sequence)

    Sparse sequence record:
    - 

  */

  sequence_memory_dataset<finite_dataset>;
  sequence_memory_dataset<vector_dataset<> >;
  sequence_memory_dataset<finite_variable>;
  sequence_memory_dataset<vector_variable>;

  sequence_dataset<finite_variable>;
  sequence_dataset<finite_dataset>;
  sequence_dataset<vector_dataset<> >

  typedef sequence_dataset<typename Factor::dataset_type> dataset_type;
  typedef sequence_dataset<variable_type, elem_type> dataset_type;
  sequence_record<variable_type> 


  


  
  sequence_dataset<finite_dataset> dataset;
  sequence_record<finite_variable> record;

  sequence_record<finite_variable> 

  class opaque_data { };

  class finite_data {
    size_t vector_size() { return 1; }
    size_t get_index(v) { return v->index(); }
    T* ptr(size_t index, size_t offset) { }
  };

  class vector_data { };
  
  // can use raw_record_iterator_state or similar (pointers to void*)!!!
  // requirement: trivial constructor

  // alternative: include sequence_value

  };




    
}
