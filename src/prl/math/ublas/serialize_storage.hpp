#ifndef PRL_MATH_UBLAS_SERIALIZE_STORAGE_HPP
#define PRL_MATH_UBLAS_SERIALIZE_STORAGE_HPP

#include <boost/numeric/ublas/storage.hpp>

#include <prl/macros_def.hpp>

// A few additional serialization functions
namespace boost { namespace numeric { namespace ublas
{

  //! \ingroup math_ublas
  template<class Archive, class Z, class D>
  inline void serialize(Archive& ar, 
			basic_range<Z,D>& range, 
			const unsigned int /* file_version */) {
    Z start = range.start(), stop = range.start() + range.size();
    ar & serialization_nvp(start);
    ar & serialization_nvp(stop);
    if (Archive::is_loading::value) 
      range = basic_range<Z,D>(start, stop);
  }
  
} } } // namespaces

#include <prl/macros_undef.hpp>

#endif
