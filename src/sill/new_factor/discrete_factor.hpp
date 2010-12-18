#ifndef SILL_DISCRETE_FACTOR_HPP
#define SILL_DISCRETE_FACTOR_HPP


#include <sill/base/finite_variable.hpp>
#include <sill/base/assignment.hpp>
#include <sill/base/finite_assignment_iterator.hpp>


#include <sill/macros_def.hpp>

namespace sill {

  class discrete_factor {
  private:
    finite_domain domain_;
    bool log_space_;

  public:

    //! Returns the arguments associated with the discrete factor
    virtual const finite_domain& domain() const {
      return domain_;
    }
    
    //! Assign a new domain to this factor
    virtual void set_domain(const finite_domain& new_domain) {
      domain_ = new_domain;
    } 

    //! Returns true if the data is stored in log space
    bool log_space() { 
      return log_space_; 
    }
    
    /**
     * Get the value stored at the particular assignment
     * (it may be in log space).
     */
    virtual double get(const finite_assignment& asg) const = 0;
    virtual void set(const finite_assignemnt& asg, double value) = 0;

    /**
     * This implements a basic binary operation of the form:
     *
     *   C(x) = this(x) * B(x)
     *
     * TODO: add logspace support
     */
    virtual discrete_factor& 
    times(const discrete_factor& b, discrete_factor& c) const {
      // Compute the union of the domains
      finite_domain union_domain;
      union_domain.insert(domaion().begin(), domain.end());
      union_domain.insert(b.domain().begin(), b.domain().end());
      // update the domain of C
      c.set_domain(union_domain);
      // Compute C = this * B
      foreach(const finite_assignment& asg, assignments(union_domain)) {
        c.set(asg, get(asg) * b.get(asg));
      }
      // Return the result
      return c;
    }


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
      // Compute the new domain of c
      finite_domain cdomain = domain();
      // cdomain = domain() setminus b.domain()
      cdomain.erase(b.domain().begin(), b.domain().end());
      // Set the new domain for c
      c.set_domain(cdomain);
      // for each entry in c(x)
      foreach(const finite_assignment& xasg, assignments(cdomain)) {
        double sum = 0.0;
        // for each entry in b(y)
        foreach(const finite_assignment& yasg, assignments(b.domain())) {
          // Construct the assignment (x,y);
          finite_assignment xyasg = xasg;
          xyasg.insert(yasg.begin(), yasg.end());
          sum += get(xyasg) * b.get(yasg);
        }
        c(xasg) = sum;
      } // end of for each entry in c(x)
      return c;
    }

    //! Normalize a factor
    virtual void normalize() {
      // Compute the sum
      double sum = 0.0;
      foreach(const finite_assignment& asg, assignments(domain())) {
        sum += get(asg);
      }
      foreach(const finite_assignment& asg, assignments(domain())) {
        set(asg, get(asg) / sum);
      }     
    }

  }; // end of discrete_factor

}; // end of namespace sill

#include <sill/macros_undef.hpp>
#endif
