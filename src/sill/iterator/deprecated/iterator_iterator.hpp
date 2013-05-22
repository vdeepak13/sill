// Probabilistic Reasoning Library (PRL)
// Copyright 2005, 2008 (see AUTHORS.txt for a list of contributors)
// 
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

#ifndef SILL_ITERATOR_ITERATOR_HPP
#define SILL_ITERATOR_ITERATOR_HPP

#include <iterator>
#include <sill/global.hpp>

namespace sill {

  /**
   * An iterator adaptor whose dereference operation returns the
   * original iterator.  This is useful for turning an iterator range
   * over a sequence of values in to an iterator range into the
   * corresponding range of iterators.  For example, to create a
   * vector of the iterators in some container C, we can write
   * 
   *   std::vector<iterator_type> v(sill::make_it_iterator(C.begin()), 
   *                                sill::make_it_iterator(C.end()));
   *
   * \todo Where is class template this used? 
   *
   * \ingroup iterator
   */
  template <typename It>
  class iterator_iterator : public std::iterator
    <typename std::iterator_traits<It>::iterator_category, 
     It, 
     typename std::iterator_traits<It>::difference_type>
  {

  protected:

    //! The underlying iterator.
    It it;

  public:

    //! Constructor.
    explicit iterator_iterator(It it) : it(it) { }

    //! Returns the underlying iterator (and not its referent).
    It operator*() const { return it; }

    //! Returns a pointer to the underlying iterator.
    It* operator->() const { return &it; }

    //! Advances the underlying iterator.
    iterator_iterator& operator++() { 
      ++(this->it);
      return *this;
    }

    //! Advances the underlying iterator.
    iterator_iterator operator++(int) { 
      iterator_iterator cur(*this);
      ++(*this);
      return cur;
    }

    //! Decrements the underlying iterator.
    iterator_iterator& operator--() { 
      --(this->it);
      return *this;
    }

    //! Decrements the underlying iterator.
    iterator_iterator operator--(int) { 
      iterator_iterator cur(*this);
      --(*this);
      return cur;
    }

    //! Equality operator.
    bool operator==(const iterator_iterator& x) const {
      return (this->it == x.it);
    }

    //! Inequality operator.
    bool operator!=(const iterator_iterator& x) const {
      return (this->it != x.it);
    }

  };

  //! A convenience function for creating iterator iterators.
  //! \relates iterator_iterator
  template <typename It>
  iterator_iterator<It> make_iterator_iterator(It it) {
    return iterator_iterator<It>(it);
  }

} // namespace sill

#endif // #ifndef SILL_ITERATOR_ITERATOR_HPP
