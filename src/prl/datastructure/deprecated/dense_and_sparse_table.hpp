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

#ifndef SILL_DENSE_AND_SPARSE_TABLE_HPP
#define SILL_DENSE_AND_SPARSE_TABLE_HPP

#include <sill/macros_def.hpp>

namespace sill {

  //! Constructs a sparse table from a dense one.
  template <typename T>
  template <typename U>
  sparse_table_t<T>::sparse_table_t (const dense_table<U>& table, T default_elt)
    : geometry(table.shape()),
      ielt_map(),
      ielt_index_vec(table.arity()),
      default_elt(default_elt)
  { 
    using namespace boost;
    function_requires< ConvertibleConcept<U, T> >();
    foreach(const typename dense_table<U>::shape_type& index, table) {
      T elt = static_cast<T>(table.get(index));
      if (elt != default_elt)
	set(index, elt);
    }
  }

  /**
   * Constructs a dense table from a sparse one.  Declared in
   * dense_table.hpp.
   */
  template <typename T>
  template <typename U>
  dense_table<T>::dense_table(const sparse_table_t<U>& table)
    : base(table.shape()), 
      offset(table.shape()), 
      elts(table_size(table.shape()), table.get_default_elt()) {
    typename sparse_table_t<U>::const_ielt_it_t it, end;
    for (boost::tie(it, end) = table.indexed_elements(); it != end; ++it) 
      operator()(it->first) = it->second;
  }

} // namespace sill

#include <sill/macros_def.hpp>

#endif // #ifndef SILL_DENSE_AND_SPARSE_TABLE_HPP
