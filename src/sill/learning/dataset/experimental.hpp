namespace sill {

  class finite_timed_dataset {

    typedef finite_timed_process    argument_type;
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
    sliding(finite_timed_process p, size_t history) const;

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

}
