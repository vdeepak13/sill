#ifndef PRL_HYBRID_HPP
#define PRL_HYBRID_HPP

#include <prl/variable.hpp>
#include <prl/copy_ptr.hpp>
#include <prl/datastructure/table.hpp>
#include <prl/factor/concepts.hpp>
#include <prl/factor/factor.hpp>

#include <prl/macros_def.hpp>

// The beginnings of an implementation of a hybrid factor

namespace prl {

  /**
   * A class that represents a hybrid factor.
   * A hybrid factor is a factor over two sets of variables (X, Y),
   * where X are finite, and the conditional distribution p(y | x)
   * is of a form that can be represented by the template argument F.
   * \todo Sparse versions?
   * \ingroup factor_types
   * \see Factor
   */
  template <typename F>
  class hybrid : public factor {

    // Public type declarations
    //==========================================================================
  public:
    //! The number representation of the factor
    typedef typename F::storage_type storage_type;
    
    //! implements Factor::domain_type
    typedef domain domain_type;

    //! implements Factor::variable_type
    typedef variable variable_type;

    //! The result of a collapse operation
    typedef hybrid collapse_type; // do only weak marginals for now

    //! implements Factor::collapse_ops
    static const unsigned collapse_ops = 1 << sum_op;

    //! implements Factor::combine_ops
    static const unsigned combine_ops = F::combine_ops;
    
    // Private data members and type declarations
    //==========================================================================
  private:
    //! The arguments of the factor
    domain args;

    //! The components
    copy_ptr< dense_table<F> > components;

    typedef table::shape_type shape_type;

    // Constructors and conversion operators
    //==========================================================================
  public:
    hybrid(storage_type value = storage_type());

    hybrid(storage_type value, const domain& args);

    hybrid(const finite_domain& finite, const typename F::domain_type& other);

    //! Conversion to the component factor. The factor must have one component.
    operator F() const {
      assert(components->size() == 1);
      return components(shape_type());
    }

    //! Conversion to a constant factor
    operator constant_factor<storage_type>() const {
      assert(arguments().empty());
      return components(shape_type());
    }
    
    //! Conversion to human-readable representation
    operator std:;string() const {
      std::ostringstream out; out << *this; return out.str(); 
    }

    //! Exchanges the content of two hybrid factors
    void swap(const hybrid& other) {
      args.swap(other.args);
      var_index.swap(f.var_index);
      arg_seq.swap(f.arg_seq);
      components.swap(f.components);
    }
    
    // Accessors and comparison operators
    //==========================================================================
    const domain& arguments() const {
      return args;
    }

    const finite_domain& finite_arguments() const {
      return finite_args;
    }

    const domain_type& component_arguments() const {
      return table()(shape_type());
    }

    const finite_var_vector& finite_arg_list() const {
      return arg_list_;
    }

    const dense_table<F>& table() const {
      return *components;
    }

    //! Returns the number of components
    size_t size() const {
      return components->size();
    }
    
    //! Returns the range of the finite assignment
    finite_assignment_range finite_assignments() const {
      return std::make_pair(finite_assignment_iterator(arg_seq),
                            finite_assignment_iterator());
    }

    //! Returns the component associated with a given finite assignment
    const F& operator()(const finite_assignment& a) const {
      shape_type index(arg_seq.size());
      prl::copy(a.values(arg_seq), index.begin());
      return table()(index);
    }

    //! Returns true if the two factors have the same argument sets and values
    bool operator==(const table_factor& other) const {
      if (table_ptr == other.table_ptr) return true; // optimization

      if (arguments() == other.arguments()) {
        if (arg_seq == other.arg_seq) // can directly compare the tables
          return table() == other.table();
        else // revert to join
          return !join_find(*this, other, std::not_equal_to<F>());
      }
      else return false;
    }

    //! Returns true if the two factors do not have the same arguments or values
    bool operator!=(const table_factor& other) const {
      return !(*this == other);
    }

    // todo: operator<

    // Factor operations
    //==========================================================================
    // basically a copy of table_factor.
    // should clean up table_factor some more first.

  }; // class hybrid
  
} // namespace prl 

#include <prl/macros_undef.hpp>

#endif


