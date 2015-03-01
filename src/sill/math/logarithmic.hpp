#ifndef SILL_LOGARITHMIC_HPP
#define SILL_LOGARITHMIC_HPP

#include <limits>
#include <iostream>
#include <cmath>
#include <functional>

#include <boost/algorithm/minmax.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/math/special_functions/log1p.hpp>

#include <sill/global.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  //! A tag that is used to indicate a log-space value.
  //! \ingroup math_number
  struct log_tag { };

  /**
   * A numeric representation which represents the real number \f$x\f$
   * using a floating point representation of \f$\log x\f$.  This
   * class is designed so that the representation of the value is
   * transparent to the user.  E.g., all operators that are defined
   * for double representations are also defined for logarithmic<double>
   * representations, and they have the same meaning.  These operators
   * can also be used with both types simultaneously, e.g., 1.0 +
   * logarithmic<double>(2.0) == 3.0.
   *
   * \todo specialize std::numeric_limits
   *
   * \todo We may be able to use boost/operators.hpp to help us generate
   *       operators for this class concisely.
   *
   * \ingroup math_number
   */
  template <typename T>
  class logarithmic {
    BOOST_STATIC_ASSERT(std::numeric_limits<T>::has_infinity);

    //! The underlying representation
    typedef T value_type;

  public:
    /**
     * The log space representation of \f$x\f$, i.e., the value
     * \f$\log x\f$.
     */
    T lv;

  public:
    //! Serialize members
    void save(oarchive & ar) const{
      ar << lv;
    }

    //! Deserialize members
    void load(iarchive & ar) {
      ar >> lv;
    }

    /**
     * Default constructor.  The value is initialized to represent
     * zero.
     */
    logarithmic() : lv(-std::numeric_limits<value_type>::infinity()) { }

    /**
     * Log-space constructor.
     *
     * @param lv the logarithm of the value this object should
     *           represent
     */
    logarithmic(const value_type& lv, log_tag) : lv(lv) { }

    /**
     * Constructor.  Note that the parameter is the value to be
     * represented, not its logarithm.  This constructor can
     * participate in automatic conversions from value_type
     * in operators.
     *
     * @param value the value this object should represent
     */
    logarithmic(const value_type& value) {
      if (value == static_cast<value_type>(0))
        lv = -std::numeric_limits<value_type>::infinity();
      else
        lv = std::log(value);
    }

    //! Conversion constructor from a different logarithmic type.
    template <typename U>
    explicit logarithmic(const logarithmic<U>& lv) : lv(lv) { }

    /**
     * Conversion out of log space.  Casting a log-space value into
     * its associated storage type computes the standard
     * representation from the log-space representation.
     *
     * @return  the value \f$x\f$, where this object represents
     *          \f$x\f$ in log-space
     */
    operator value_type() const {
      return exp(lv);
    }


    /**
     * Returns the internal representation of the value in log space.
     *
     * @return  the value \f$\log x\f$, where this object represents
     *          \f$x\f$ in log-space
     */
      value_type log_value() const {
      return lv;
    }

    /**
     * Returns the log space value representing the sum of this
     * value and the supplied log space value.
     *
     * This routine exploits a special purpose algorithm called log1p
     * that is in the C standard library.  log1p(x) computes the value
     * \f$\log(1 + x)\f$ in a numerically stable way when \f$x\f$ is
     * close to zero.  Note that
     * \[
     *  \log(1 + y/x) + \log(x) = \log(x + y)
     * \]
     * Further note that
     * \[
     *  y/x = \exp(\log y - \log x)
     * \]
     * Thus, we first compute \f$y/x\f$ stably by choosing \f$x >
     * y\f$, and then use log1p to implement the first equation.
     *
     * @param a the value \f$\log x\f$
     * @return  the value \f$\log (x + y)\f$, where this object
     *          represents \f$\log y\f$
     */
    logarithmic operator+(const logarithmic& a) const {
      using namespace boost::math;
      if (a.lv == -std::numeric_limits<value_type>::infinity())
        return *this;
      if (lv == -std::numeric_limits<value_type>::infinity())
        return a;
      value_type lx, ly;
      boost::tie(ly, lx) = boost::minmax(lv, a.lv);
      return logarithmic(log1p(exp(ly - lx)) + lx, log_tag());
    }


    /**
     * Returns the log space value representing the difference of this
     * value and the supplied log space value. This works by converting
     * into real-space and taking log on the result.
     */
    logarithmic operator-(const logarithmic& a) const {
      if (lv <= a.lv) {
        // result is negative
        //TODO: ?error? return log(0)=-inf
        return logarithmic(-std::numeric_limits<value_type>::infinity(), 
                          log_tag());
      }
      else {
        return logarithmic(T(*this) - T(a));
      }
    }
    /**
     * Computes the sum of this (log-space) value and the supplied
     * value.
     *
     * @param  y the value \f$y\f$ to be added to this (log-space)
     *           value
     * @return a log-space representation of \f$x + y\f$, where this
     *           object represents \f$x\f$ in log-space
     */
    logarithmic operator+(const value_type& y) const {
      return *this + logarithmic(y);
    }

    /**
     * Returns the log space value representing the difference of this
     * value and the supplied value. This works by converting
     * into real-space and taking log on the result.
     */
    logarithmic operator-(const value_type& y) const {
      return logarithmic(T(*this) - y);
    }

    /**
     * Updates this object to represent the sum of this value and the
     * supplied value.  This method works with standard values and
     * values in log-space.
     *
     * @see operator+
     *
     * @param a the value \f$y\f$
     * @return  this value, after it has been updated to represent
     *          \f$x + y\f$ in log-space, where this object originally
     *          represented \f$x\f$ in log-space
     */
    logarithmic& operator+=(const logarithmic& y) {
      *this = *this + y;
      return *this;
    }


    /**
     * Returns the log space value representing the difference of this
     * value and the supplied log space value. This works by converting
     * into real-space and taking log on the result.
     */
    logarithmic& operator-=(const logarithmic& y) {
      *this = *this - y;
      return *this;
    }

    /**
     * Returns the value representing the product of this value and
     * the supplied value.
     *
     * @param a the value \f$y\f$ represented in log-space
     * @return  the value \f$x \times y\f$ represented in log-space,
     *          where this object represents \f$x\f$ in log-space
     */
    logarithmic operator*(const logarithmic& a) const {
      return logarithmic(this->lv + a.lv, log_tag());
    }

    /**
     * Returns the value representing the product of this value and
     * the supplied value.
     *
     * @param y the value \f$y\f$
     * @return  the value \f$x \times y\f$ represented in log-space,
     *          where this object represents \f$x\f$ in log-space
     */
    logarithmic operator*(const value_type& y) const {
      return (*this) * logarithmic(y);
    }

    /**
     * Updates this object to represent the product of this value and
     * the supplied value.  This method works with standard values and
     * values in log-space.
     *
     * @see operator*
     *
     * @param a the value \f$y\f$
     * @return  this value, after it has been updated to represent
     *          \f$x \times y\f$ in log-space, where this object
     *          originally represented \f$x\f$ in log-space
     */
    logarithmic& operator*=(const logarithmic& y) {
      (*this) = (*this) * y;
      return *this;
    }

    /**
     * Returns the value representing the ratio of this value and the
     * supplied value.
     *
     * @param a the value \f$y\f$ represented in log-space
     * @return  the value \f$x / y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    logarithmic operator/(const logarithmic& a) const {
      return logarithmic(this->lv - a.lv, log_tag());
    }

    /**
     * Returns the value representing the ratio of this value and the
     * supplied value.
     *
     * @param y the value \f$y\f$
     * @return  the value \f$x / y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    logarithmic operator/(const value_type& y) const {
      return (*this) / logarithmic(y);
    }

    /**
     * Updates this object to represent the ratio of this value and
     * the supplied value.  This method works with standard values and
     * values in log-space.
     *
     * @see operator/
     *
     * @param a the value \f$y\f$
     * @return  this value, after it has been updated to represent
     *          \f$x \times y\f$ in log-space, where this object
     *          originally represented \f$x\f$ in log-space
     */
    template <typename U>
    logarithmic& operator/=(const U& y) {
      (*this) = (*this) / y;
      return *this;
    }

    /**
     * Returns true iff this object represents the same value as the
     * supplied object.
     *
     * @param a the value \f$y\f$ represented in log-space
     * @return  true if \f$x = y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    bool operator==(const logarithmic& a) const {
      return (lv == a.lv);
    }

    /**
     * Returns true iff this object represents a different value from
     * the supplied object.
     *
     * @param a the value \f$y\f$ represented in log-space
     * @return  true if \f$x \neq y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    bool operator!=(const logarithmic& a) const {
      return (lv != a.lv);
    }

    /**
     * Returns true iff this object represents a smaller value than
     * the supplied object.
     *
     * @param a the value \f$y\f$ represented in log-space
     * @return  true if \f$x < y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    bool operator<(const logarithmic& a) const {
      return (lv < a.lv);
    }

    /**
     * Returns true iff this object represents a larger value than
     * the supplied object.
     *
     * @param a the value \f$y\f$ represented in log-space
     * @return  true if \f$x > y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    bool operator>(const logarithmic& a) const {
      return (lv > a.lv);
    }

    /**
     * Returns true iff this object represents a value that is less
     * than or equal to the supplied object.
     *
     * @param a the value \f$y\f$ represented in log-space
     * @return  true if \f$x \le y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    bool operator<=(const logarithmic& a) const {
      return (lv <= a.lv);
    }

    /**
     * Returns true iff this object represents a value that is greater
     * than or equal to the supplied object.
     *
     * @param a the value \f$y\f$ represented in log-space
     * @return  true if \f$x \ge y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    bool operator>=(const logarithmic& a) const {
      return (lv >= a.lv);
    }

    /**
     * Returns (this>0 && a>0)
     */
    bool operator&&(const logarithmic& a) const {
      return (lv != -std::numeric_limits<value_type>::infinity() 
              && 
              a.lv != -std::numeric_limits<value_type>::infinity());
    }
    
    /**
     * Returns (this>0 || a>0)
     */
    bool operator||(const logarithmic& a) const {
      return (lv != -std::numeric_limits<value_type>::infinity() 
              || 
              a.lv != -std::numeric_limits<value_type>::infinity());
    }
    /**
     * Writes this log space representation to the supplied stream.
     */
    void write(typename std::ostream& out) const {
      out << "exp(" << lv << ")";
    }

    /**
     * Reads this log space value from the supplied stream.  There are
     * two accepted formats.  The first the same format used by the
     * value_type type; numbers in this format are converted into log
     * space representation.  The second format is 'exp(X)', where X
     * is in a format used by the value_type type.  In this case, the
     * read value is treated as a log space value.  For example,
     * reading '1.23e4' causes this object to represent the value
     * 1.23e4 in log space, as log(1.23e4); reading the value
     * exp(-1234.5) causes this object to represent the value
     * \f$e^{-1234.5}\f$, by storing the value 1234.5.
     *
     * @param in the stream from which this value is read
     */
    void read(typename std::istream& in) {
      // Read off any leading whitespace.
      in >> std::ws;
      // Check to see if this value is written in log space.
      typedef typename std::istream::int_type int_t;
      if (in.peek() == static_cast<int_t>('e')) {
        in.ignore(4);
        in >> lv;
        in.ignore(1);
      } else {
        value_type x;
        in >> x;
        *this = x;
      }
    }

  }; // struct logarithmic

  /**
   * Logarithmic value with double storage.
   */
  typedef logarithmic<double> logd;

  /**
   * Returns the value representing the sum of this value and the
   * supplied value.
   *
   * @param x the value \f$x\f$
   * @param a an object representing \f$y\f$ in log-space
   * @return  the value \f$x + y\f$ represented in log-space
   *
   * \relates logarithmic
   */
  template <typename T>
  inline logarithmic<T> operator+(const T& x, const logarithmic<T>& a) {
    return a + x;
  }

  /**
   * Returns the value representing the product of this value and
   * the supplied value.
   *
   * @param x the value \f$x\f$
   * @param a an object representing \f$y\f$ in log-space
   * @return  the value \f$x \times y\f$ represented in log-space
   *
   * \relates logarithmic
   */
  template <typename T>
  inline logarithmic<T> operator*(const T& x, const logarithmic<T>& a) {
    return a * x;
  }

  /**
   * Returns the value representing the ratio of this value and
   * the supplied value.
   *
   * @param x the value \f$x\f$
   * @param a an object representing \f$y\f$ in log-space
   * @return  the value \f$x / y\f$ represented in log-space
   *
   * \relates logarithmic
   */
  template <typename T>
  inline logarithmic<T> operator/(const T& x, const logarithmic<T>& a) {
    return logarithmic<T>(x) / a;
  }
  
  //! Returns the power of the logarithmic object
  //! \relates logarithmic
  template <typename T>
  inline logarithmic<T> pow(const logarithmic<T>& x, double s) {
    return logarithmic<T>(x.log_value() * s,log_tag());
  }
  /**
   * Writes a log space value to a stream.
   * \relates logarithmic
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const logarithmic<T>& x) {
    x.write(out);
    return out;
  }

  /**
   * Reads a log space value from a stream.
   * \relates logarithmic
   */
  template <typename T>
  std::istream& operator>>(std::istream& in, logarithmic<T>& x) {
    x.read(in);
    return in;
  }

} // namespace sill

namespace std {
  // overload the log in std
  //! Returns the log-value of the logarithmic object
  //! \relates logarithmic
  template <typename T>
  inline T log(const sill::logarithmic<T>& x) {
    return x.log_value();
  }
}

namespace sill{ 
  using std::log;
}
#include <sill/macros_undef.hpp>

#endif // SILL_LOGARITHMIC_HPP



