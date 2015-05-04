#ifndef SILL_VARIABLE_HPP
#define SILL_VARIABLE_HPP

#include <sill/argument/argument_object.hpp>
#include <sill/argument/basic_domain.hpp>
#include <sill/functional/hash.hpp>

#include <unordered_map>

namespace sill {

  // Forward declaration
  template <typename Index, typename Variable> class process;
  class universe;

  /**
   * A class that represents a basic variable. Internally, a variable
   * consists of a pointer to the argument object and time index 
   * (which can be empty). The argument object is shared among the
   * variable copies and must persist past the lifetime of this one.
   * Variables are not created directly; instead, they are created
   * through the universe class.
   *
   * This class models the Argument concept.
   */
  class variable {
  public:
    typedef argument_object::category_enum category_enum;

    //! Constructs an empty variable.
    variable()
      : rep_(nullptr), index_(-1) { }

    //! Converts the variable to bool indicating if the variable is empty.
    explicit operator bool() const {
      return rep_ != nullptr;
    }

    //! Saves the variable to an archive.
    void save(oarchive& ar) const {
      ar.serialize_dynamic(rep_);
      ar << index_;
    }

    //! Loads the variable from an archive.
    void load(iarchive& ar) {
      rep_ = ar.deserialize_dynamic<argument_object>();
      ar >> index_;
    }

    //! Returns the cardinality / dimensionality of the variable.
    size_t size() const {
      return rep().size;
    }

    //! Returns the index of the variable (for now, a size_t).
    size_t index() const {
      return index_;
    }

    //! Returns true if the variable is associated with a process.
    bool indexed() const {
      return index_ != size_t(-1);
    }

    //! Returns the category of the variable (finite / vector).
    category_enum category() const {
      return rep().category;
    }

    //! Returns true if the variable is finite.
    bool finite() const {
      return rep().category == argument_object::FINITE;
    }

    //! Returns true if the variable is vector.
    bool vector() const {
      return rep().category == argument_object::VECTOR;
    }

    //! Returns the name of the variable.
    const std::string& name() const {
      return rep().name;
    }

    //! Conversion to human-readable representation.
    std::string str() const {
      return indexed()
        ? rep().str() + '(' + std::to_string(index_) + ')'
        : rep().str();
    }

    //! Compares two variables.
    friend bool operator==(variable x, variable y) {
      return x.rep_ == y.rep_ && x.index_ == y.index_;
    }

    //! Compares two variables.
    friend bool operator!=(variable x, variable y) {
      return x.rep_ != y.rep_ || x.index_ != y.index_;
    }

    //! Compares two variables.
    friend bool operator<(variable x, variable y) {
      return std::make_pair(x.rep_, x.index_) < std::make_pair(y.rep_, y.index_);
    }

    //! Compares two variables.
    friend bool operator>(variable x, variable y) {
      return std::make_pair(x.rep_, x.index_) > std::make_pair(y.rep_, y.index_);
    }

    //! Returns true if two variables are type-compatible.
    friend bool compatible(variable x, variable y) {
      return x.size() == y.size() && x.category() == y.category();
    }

    //! Computes the hash of the variable.
    friend size_t hash_value(variable x) {
      size_t seed = 0;
      sill::hash_combine(seed, x.rep_);
      sill::hash_combine(seed, x.index_);
      return seed;
    }

    //! Prints a variable to an output stream.
    friend std::ostream& operator<<(std::ostream& out, variable x) {
      out << x.rep();
      if (x.indexed()) out << '(' << x.index() << ')';
      return out;
    }

  private:
    //! Constructs a free variable with the given argument object.
    explicit variable(const argument_object* rep)
      : rep_(rep), index_(-1) { }

    //! Constructs a proces variable with the given argument object and index.
    variable(const argument_object* rep, size_t index)
      : rep_(rep), index_(index) { }

    //! Returns a reference to the underlying argument object.
    const argument_object& rep() const {
      assert(rep_ != nullptr);
      return *rep_;
    }
    
    //! The underlying representation.
    const argument_object* rep_;

    //! The index associated with the variable.
    size_t index_;

    // Friends
    template <typename Index, typename Var> friend class process;
    friend class universe;
    friend class std::hash<variable>;

  }; // class variable

  //! A type that represents a domain of variables.
  typedef basic_domain<variable> domain;

  //! A type that maps one variable to another.
  typedef std::unordered_map<variable, variable> variable_map;

} // namespace sill


namespace std {

  template <>
  struct hash<sill::variable> {
    typedef sill::variable argument_type;
    typedef size_t result_type;
    size_t operator()(sill::variable x) const {
      return hash_value(x);
    }
  };

} // namespace std

#endif
