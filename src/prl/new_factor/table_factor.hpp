#ifndef PRL_TABLE_FACTOR_HPP
#define PRL_TABLE_FACTOR_HPP

#include <vector>
#include <typeinfo>

#include <prl/base/finite_variable.hpp>
#include <prl/base/assignment.hpp>
#include <prl/base/finite_assignment_iterator.hpp>


#include <prl/macros_def.hpp>

namespace prl {

  class table_factor : public discrete_factor {
  private:
    typedef discrete_factor base;

    std::vector<double> data_;

  public:
    
    //! Assign a new domain to this factor
    virtual void set_domain(const finite_domain& new_domain) {
      // set the base domain value
      base::set_domain(new_domain);
      // compute the size of the table
      typedef finite_domain::value_type;
      size_t table_size = 1;
      foreach(value_type& pair, new_domain) {
        table_size *= pair.first->size();
      }
      // resize the data table
      data_.resize(table_size);
    } // end of set_domain


    //! make the factor sum to 1;
    void make_uniform() {
      double value = 1/data_.size();
      for(size_t i = 0; i < data_.size(); ++i) {
        data_[i] = value;
      }
    } // end of make_uniform

    
    /**
     * Compute the mapping from the finite domain to index
     */
    size_t asg2ind(const finite_assignment& asg) const {
      typedef finite_assignment::const_iterator iterator;
      size_t index = 0; 
      size_t offset = 1;
      foreach(const finite_variable* v, domain()) {
        iterator iter = asg.find(v);
        assert(iter != asg.end());
        index += iter->second * offset;
        offset *= v->size();
      }
      return index;
    } // end of asg2ind


    virtual double get(const finite_assignment& asg) const {
      size_t index = asg2ind(asg);
      assert(index < data_.size());
      return data_[index];
    } // end of get


    virtual void set(const finite_assignemnt& asg, double value) 
    {
      size_t index = asg2ind(asg);
      assert(index < data_.size());
      data_[index] = value;
    } // end of set


    /**
     * This implements a basic binary operation of the form:
     *
     *   C(x) = this(x) * B(x)
     *
     * TODO: add logspace support
     */
    virtual discrete_factor& 
    times(const discrete_factor& b, discrete_factor& c) const {
      
      if(typeid(b) == typeid(table_factor) &&
         typeid(c) == typeid(table_factor) ) {
        // do some optimal implementation
      } else {
        // Call the base implementation
        return base::times(b,c);
      }
      // Return the result
      return c;
    } // end of times


    /**
     * This implements a basic binary operation of the form:
     *
     *   C(x) = Sum_{y in domain(B)} A(x,y) * B(y)
     *
     * TODO: make log_space aware. This could be made more efficient
     * if we had a c.zero() function so we would only require a single
     * loop
     */
    virtual discrete_factor&
    prodmarg(const discrete_factor& b, discrete_factor& c) const {
      
      if(typeid(b) == typeid(const table_factor&) &&
         typeid(c) == typeid(const table_factor&) ) {
        return prodmarg(dynamic_cast<const table_factor&>(b), 
                        dynamic_cast<const table_factor&>(c)); 
      } else {
        // Call the base implementation
        return base::prodmarg(b,c);
      }
      // Return the result
      return c;
    } // end of prodmarg

    
    discrete_factor&
    prodmarg(const table_factor& b, table_factor& c) const {
      // Do something optimal
      return c;
    } // end of prodmarg


    //! Normalize a factor
    virtual void normalize() {
      // Compute the sum
      double sum = 0.0;
      for(size_t i = 0; i < data_.size(); ++i) {
        sum += data_[i]; 
      }
      for(size_t i = 0; i < data_.size(); ++i) {
        data_[i] = data_[i] / sum; 
      }
    } // end of normalize


  }; // end of discrete_factor

}; // end of namespace prl

#include <prl/macros_undef.hpp>
#endif


#endif
