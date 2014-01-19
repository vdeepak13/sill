#ifndef SILL_ANY_FACTOR_HPP
#define SILL_ANY_FACTOR_HPP

#include <iosfwd>
#include <typeinfo>

#include <sill/factor/factor.hpp>
#include <sill/factor/constant_factor.hpp>
#include <sill/factor/any_factor_placeholder.hpp>
#include <sill/factor/any_factor_binary.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that provides type erasure for factors.
   *
   * In certain cases, it is useful to allow multiple types of factors
   * be held in a container. In these cases, any_factor can be used to
   * to store and manipulate factors of arbitrary types. This class
   * serves as a wrapper that can hold any factor that subclasses
   * sill::factor.
   *
   * Initially, the user needs to register all the types that can be
   * held by this factor class, as well as the binary operations among
   * the allowed factor classes. Standard classes, including table_factor,
   * moment_gaussian, and canonical_gaussian, are registered automatically.
   * 
   * Implementation note: Before any mutating operation that changes
   * the data pointed to by wrapper, you _must_ call
   * ensure_unique(). Most operations only require one virtual
   * function call. Combine operations and assignment / construction
   * from the abstract base class also require RTTI and a single map
   * lookup. (Binary operations currently require more than one
   * virtual function call, but the performance could be optimized
   * here). The factors, held by this class, are always allocated on
   * the heap.
   *
   * \ingroup factor_types
   * \see Factor
   */
  class any_factor : public factor {

    // Public type declaration
    //==========================================================================
  public:
    //! The base class
    typedef factor base;

    //! implements Factor::result_type
    typedef double result_type; // maybe should be logarithmic<double>

    //! implements Factor::domain_type
    typedef domain domain_type;

    //! implements Factor::variable_type
    typedef variable variable_type;
    
    //! The result of a collapse operation
    typedef any_factor collapse_type;

    //! implements Factor::collapse_ops
    static const unsigned collapse_ops = ~0; // supports all operations

    //! implements Factor::combine_ops
    static const unsigned combine_ops = ~0; // supports all operations

  private:

    //! A pair of type_info pointers
    typedef std::pair<const std::type_info*, const std::type_info*> 
      info_ptr_pair;
    
    //! A comparator for type_info classes
    //! MSVC's type_info::before returns an int, rather than bool
    //! so we return int here to supress warnings
    struct type_info_less {
      int operator()(const std::type_info* a, const std::type_info* b) const {
        return a->before(*b);
      }

      int operator()(const info_ptr_pair& a, const info_ptr_pair& b) const {
        if (*a.first == *b.first)
          return a.second->before(*b.second);
        else
          return a.first->before(*b.first);
      }
    };

    // Private data members and helper functions
    //==========================================================================
  private:
    //! A map from factor type to the corresponding polymorphic wrapper
    typedef std::map<const std::type_info*, factor_placeholder*, type_info_less>
      factor_map;

    //! A map from a pair of factor types to a polymorphic binary wrapper
    typedef std::map<info_ptr_pair, factor_binary*, type_info_less> binary_map;

    //! A registry of factor types
    static factor_map factor_registry;

    //! A registry of binary operations
    static binary_map binary_registry;

    //! A shared pointer to a wrapper around the underlying factor
    boost::shared_ptr<factor_placeholder> wrapper;

    //! The arguments of this factor
    domain_type args;

    friend class boost::serialization::access;

    //! Serialize / deserialize members
    template <class Archive>
    void serialize(Archive& ar, const unsigned int file_version);
    
    //! Makes a defensive copy of the factor if the owner is not unique
    void ensure_unique() {
      if (!wrapper.unique()) wrapper.reset(wrapper->clone());
    }

    //! Creates a new wrapper for a given factor type.
    //! The type must be registered
    static factor_placeholder* new_wrapper(const factor& f);

    // Factor registration
    //==========================================================================
  public:
    /**
     * Registers a factor type and the binary operations on a pair (F, F)
     * The factor class & XML header must be included before the factor
     * is registered
     */
    template <typename F>
    static void register_factor() {
      factor_registry[&typeid(F)] = new factor_wrapper<F>();
      register_binary<F, F>();
    }

    //! Returns true if the given factor type has been registered
    template <typename F>
    static bool registered() {
      return factor_registry.find(&typeid(F)) != factor_registry.end();
    }

    //! Registers a combine operation.
    //! Automatically registers both combine(F, G) and combine(G, F)
    template <typename F, typename G>
    static void register_binary() {
      info_ptr_pair key_fg(&typeid(F), &typeid(G));
      info_ptr_pair key_gf(&typeid(G), &typeid(F));
      binary_registry[key_fg] = new binary_wrapper<F, G>();
      binary_registry[key_gf] = new binary_wrapper<G, F>(); 
      // if key_fg == key_gf, will overwrite the map entry, but that's ok
    }

    //! Returns the binary operation wrapper for a given pair of factors
    static const factor_binary& binary(const factor& x, const factor& y);

    //! Returns the binary operation wrapper for a given pair of factors
    static const factor_binary& binary(const any_factor& x,const any_factor& y){
      return binary(x.get(), y.get());
    }

    // Constructors and conversion operators
    //==========================================================================
  public:
    //! Initializes to the given constant
    any_factor(double value = 0.0) 
      : wrapper(new factor_wrapper<constant_factor>(value)) {}

    //! Initializes to the given factor
    //! \require the factor type referenced by f must be registered
    explicit any_factor(const factor& f)
      : wrapper(new_wrapper(f)), args(wrapper->arguments()) {}
  
    //! Constructs factor with the given wrapper
    //! This constructor is invoked in standard factor operations, e.g. combine
    any_factor(factor_placeholder* wrapper) 
      : wrapper(wrapper), args(wrapper->arguments()) {}

    //! Assigns a factor to this object
    //! \require the factor type referenced by f must be registered
    any_factor& operator=(const factor& f);

    //! Conversion to human-readable representation
    operator std::string() const;

    // Accessors
    //==========================================================================
    //! Returns the argument set of this factor
    const domain_type& arguments() const {
      return args;
    }

    //! Returns the underlying factor
    const factor& get() const {
      return wrapper->get();
    }

    //! Returns the underlying factor, converted to the specified type
    template <typename Factor>
    Factor get() const {
      BOOST_STATIC_ASSERT((boost::is_base_of<factor, Factor>::value));
      Factor result;
      binary(get(), result).convert(get(), result); 
      return result;
    }    

    //! Returns a new copy of the underlying factor
    factor* copy() const {
      return wrapper->copy();
    }

    //! Returns true if two factors are of the same type and equivalent
    bool operator==(const any_factor& g) const;

    //! Returns true if the two factors are of different type or not equivalent
    bool operator!=(const any_factor& g) const {
      return !operator==(g);
    }

    //! Returns true if this factor precedes the other one
    bool operator<(const any_factor& g) const;

    // Factor operations
    //==========================================================================
    //! Evaluates the factor
    double operator()(const assignment& a) const {
      return wrapper->operator()(a);
    }

    //! Combines two factors
    //! \require the factor types must be registered
    static any_factor combine_(const factor& x, const factor& y, op_type op) {
      return binary(x, y).combine_(x, y, op);
    }

    //! implements Factor::combine_in
    any_factor& combine_in(const any_factor& other, op_type op) {
      // For now, no more efficient than combine
      *this = combine_(get(), other.get(), op);
      return *this;
    }

    //! implements Factor::combine_in
    any_factor& combine_in(const factor& other, op_type op) {
      *this = combine_(get(), other, op);
      return *this;
    }

    //! implements Factor::collapse
    any_factor collapse(const domain& retained, op_type op) const {
      return wrapper->collapse(retained, op);
    }

    //! implements Factor::restrict
    any_factor restrict(const assignment& a) const {
      return wrapper->restrict(a);
    }

    //! implements Factor::subst_args
    any_factor& subst_args(const var_map& var_map);

    //! implements DistributionFactor::marginal
    any_factor marginal(const domain& retain) const {
      return wrapper->collapse(retain, sum_op);
    }

    //! implements Factor::maximum
    any_factor maximum(const domain& retain) const { 
      return wrapper->collapse(retain, max_op);
    }
    
    //! implements Factor::minimum
    any_factor minimum(const domain& retain) const {
      return wrapper->collapse(retain, min_op);
    }

    //! implements DistributionFactor::norm_constant
    double norm_constant() const {
      return wrapper->norm_constant();
    }

    //! implements DistributionFactor::is_normalizable
    bool is_normalizable() const {
      return wrapper->is_normalizable();
    }

    //! implements DistributionFactor::normalize
    any_factor& normalize();

    //! implements DistributionFactor::entropy
    double entropy() const {
      assert(false); return 0; // wrapper->entropy();
    }

    //! implements DistributionFactor::relative_entropy
    double relative_entropy(const any_factor& other) const {
      assert(false); return 0;
    }

    //! implements Factor::arg_max
    assignment arg_max() const {
      return wrapper->arg_max_();
    }
    
    //! implements Factor::arg_min
    assignment arg_min() const {
      return wrapper->arg_min_();
    }

    //! Helper function
    void save(xml_oarchive& out) const {
      wrapper->save(out);
    }

  };

  //! \relates any_factor
  inline std::ostream& operator<<(std::ostream& out, const any_factor& f) {
    out << std::string(f);
    return out;
  }

  // Factor combinations
  //============================================================================

  //! Combines two polymorphic factors
  //! \relates any_factor
  inline any_factor 
  combine(const any_factor& x, const any_factor& y, op_type op) {
    return any_factor::combine_(x.get(), y.get(), op);
  }

  //! Combines a polymorphic factor and an arbitrary factor
  //! \pre the type referenced to by y must be registered
  //! \relates any_factor
  inline any_factor combine(const any_factor& x, const factor& y, op_type op) {
    return any_factor::combine_(x.get(), y, op);
  }

  //! Combines a polymorphic factor and an arbitrary factor
  //! \pre the type referenced to by x must be registered
  //! \relates any_factor
  inline any_factor combine(const factor& x, const any_factor& y, op_type op) {
    return any_factor::combine_(x, y.get(), op);
  }

  template<> struct combine_result<any_factor, any_factor> {
    typedef any_factor type;
  };

  template<typename F>
  struct combine_result<any_factor, F> {
    typedef any_factor type;
  };

  template<typename F>
  struct combine_result<F, any_factor> {
    typedef any_factor type;
  };

  // Need to define these to avoid ambiguity
  template<> struct combine_result<constant_factor, any_factor> {
    typedef any_factor type;
  };

  template<> struct combine_result<any_factor, constant_factor> {
    typedef any_factor type;
  };

  // Other binary operations
  //============================================================================

  //! Computes the L1 norm of the difference of two polymorphic factors
  //! \relates any_factor
  inline double norm_1(const any_factor& x, const any_factor& y) {
    return any_factor::binary(x, y).norm_1_(x.get(), y.get());
  }

  //! Computes the L-infinity norm of the difference of two table factors
  //! \relates any_factor
  inline double norm_inf(const any_factor& x, const any_factor& y) {
    return any_factor::binary(x, y).norm_inf_(x.get(), y.get());
  }

  //! Computes (1-alpha)*f1+alpha*f2 
  //! \relates any_factor
  inline any_factor 
  weighted_update(const any_factor& x, const any_factor& y, double a){
    return any_factor::binary(x, y).weighted_update_(x.get(), y.get(), a);
  }

  //! Returns true if two factors are of the same type and equivalent
  //! \relates any_factor
  inline bool operator==(const factor& f, const any_factor& g) {
    return g == any_factor(f); // invoke the member function
  }

  //! \relates any_factor
  inline bool operator==(const any_factor& f, const factor& g) {
    return f == any_factor(g);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
