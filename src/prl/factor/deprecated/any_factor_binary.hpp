#ifndef SILL_ANY_FACTOR_BINARY_HPP
#define SILL_ANY_FACTOR_BINARY_HPP

#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>

#include <sill/factor/any_factor_placeholder.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * An interface that provides type erasure for binary factor operations.
   * Each clas that implements this interface is designed for a pair of 
   * factor types and statically casts its arguments to the correct types.
   */
  struct factor_binary {
    virtual void 
    convert(const factor& x, factor& y) const = 0;

    virtual factor_placeholder*
    combine_(const factor& x, const factor& y, op_type op) const = 0;

    virtual double 
    norm_1_(const factor& x, const factor& y) const = 0;

    virtual double
    norm_inf_(const factor& x, const factor& y) const = 0;

    virtual factor_placeholder*
    weighted_update_(const factor& x, const factor& y, double a) const = 0;
    
    virtual ~factor_binary() {}

  }; // interface factor_binary


  /**
   * A polymorphic wrapper for binary operations.
   * All functions in this class require that x and y are of type
   * F and G, respectively.
   */
  template <typename F, typename G>
  class binary_wrapper : public factor_binary {
    typedef typename combine_result<F, G>::type result_type;
    typedef factor_wrapper<result_type> wrapper_type;
    
    //! Statically casts a factor to the given type
    template <typename Result>
    static const Result& cast(const factor& f) { 
      return static_cast<const Result&>(f);
    }

    //! Statically casts a factor to the given type
    template <typename Result>
    static Result& cast(factor& f) {
      return static_cast<Result&>(f);
    }

    //! Casts a factor to the result type,
    template <typename Src>
    static const result_type&
    convert_from(const factor& f, 
      typename boost::enable_if<boost::is_same<Src, result_type> >::type* = 0) {
      return cast<Src>(f);
    }

    //! Converts a factor to the result type
    template <typename Src>
    static result_type
    convert_from(const factor& f,
      typename boost::disable_if<boost::is_same<Src, result_type> >::type* =0) {
      return result_type(cast<Src>(f));
    }

  public:
    void convert(const factor& x, factor& y) const {
      cast<G>(y) = cast<F>(x);
      // for now, assume we can perform an implicit conversion to G
      // writing G(...) caused problems when the conversion operator
      // to G ought to be invoked
    }

    factor_placeholder* 
    combine_(const factor& x, const factor& y, op_type op) const {
      return new factor_wrapper<result_type>
        (combine(cast<F>(x),cast<G>(y),op));
    }

    factor_placeholder*
    weighted_update_(const factor& x, const factor& y, double a) const {
      using namespace impl; // bring the defaults into the lookup
      return new factor_wrapper<result_type>
        (weighted_update(convert_from<F>(x), convert_from<G>(y), a));
    }

    double norm_1_(const factor& x, const factor& y) const {
      using namespace impl;
      return norm_1(convert_from<F>(x), convert_from<G>(y));
    }
  
    double norm_inf_(const factor& x, const factor& y) const {
      using namespace impl;
      return norm_inf(convert_from<F>(x), convert_from<G>(y));
    }

  }; // class 

  // Default implementations of standard free functions (throw exceptions)
  namespace impl {
    
    //! The default implementation of weighted update (throws exception)
    template <typename F>
    F weighted_update(const F& x, const F& y, double a) {
      throw std::invalid_argument("Weighted update is not supported for type " +
                                  std::string(typeid(F).name()));
    }
    
    //! The default implementation of L1 norm (throws exception)
    template <typename F>
    double norm_1(const F& x, const F& y) {
      throw std::invalid_argument("norm_1 is not supported for type " +
                                  std::string(typeid(F).name()));
    }
    
    //! The default implementation of L-infinity norm (throws exception)
    template <typename F>
    double norm_inf(const F& x, const F& y) {
      throw std::invalid_argument("norm_inf is not supported for type " + 
                                  std::string(typeid(F).name()));
    }

  } // namespace impl

} // namespace sill

#include <sill/macros_undef.hpp>

#endif


