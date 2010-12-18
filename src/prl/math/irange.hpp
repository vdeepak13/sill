#ifndef SILL_INDEX_RANGE_HPP
#define SILL_INDEX_RANGE_HPP

#include <iostream>

#include <sill/serialization/serialize.hpp>

#include <sill/global.hpp>

namespace sill {

  /**
   * An open range of integral indices.
   * \ingroup math_linalg
   */
  class irange { 

  private:
    //! The left side of the range
    size_t start_;

    //! The right side of the range
    size_t stop_;
    
  public:
      
    void serialize(oarchive & ar) const{
      ar << start_ << stop_;
    }
    void deserialize(iarchive & ar) {
      ar >> start_ >> stop_;
    }

    //! Creates an empty range [0; 0)
    irange() : start_(), stop_() { }

    //! Creates an open range [start; stop)
    irange(size_t start, size_t stop) : start_(start), stop_(stop) {
      assert(start <= stop);    
    }

    //! Returns the start of the range
    size_t start() const {
      return start_;
    }

    //! Returns the end of the range
    size_t stop() const {
      return stop_;
    }

    //! Returns the end fo the range in the Matlab format, i.e., stop() - 1
    int end() const {
      return int(stop_) - 1;
    }
    
    //! Returns the size of the range
    size_t size() const { 
      return stop_ - start_;
    }

    //! Returns true if the range is empty
    bool empty() const {
      return start_ == stop_;
    }

    //! Returns the i-th element of the range
    size_t operator()(size_t i) const {
      assert(i < size());
      return start_ + i;
    }
    
  }; // class irange

  //! \relates irange
  inline std::ostream& operator<<(std::ostream& out, const irange& r) {
    out << '[' << r.start() << "; " << r.stop() << ')';
    return out;
  }
  
} // namespace sill

#endif
