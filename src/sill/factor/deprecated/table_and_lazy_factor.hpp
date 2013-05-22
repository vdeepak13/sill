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

#ifndef SILL_TABLE_AND_LAZY_FACTOR_HPP
#define SILL_TABLE_AND_LAZY_FACTOR_HPP

//! \todo Move #ifdefs here; clean up.

#include <sill/factor/table_factor.hpp>
#include <sill/factor/lazy_factor.hpp>

template <typename range_t,
	  template <typename value_t> class Table>
template <typename other_factor_t,
	  typename csr_tag_t,
	  typename elim_strategy_t,
	  typename binary_op_tag_t>
sill::table_factor<range_t, Table>::table_factor
(factor_collapse_expr<lazy_factor_t<other_factor_t,
		                      csr_tag_t,
		                      elim_strategy_t>,
                        binary_op_tag_t> expr) {
  // First collapse the lazy factor.
  lazy_factor_t<other_factor_t>
    collapsed_lazy(sill::collapse(expr.x_ptr, 
				 expr.retained, binary_op_tag_t()));

  // Flatten the lazy factor.
  sill::const_ptr_t<other_factor_t> flat_ptr = collapsed_lazy.flatten();

  // Now create a table factor which represents the flattened factor.
  table_factor result(sill::collapse(flat_ptr, 
				      expr.retained, binary_op_tag_t()));

  // Initialize this object to an empty table, and then swap it with
  // the result factor.  This avoids the overhead of copying.
  initialize(domain::empty_set, storage_t()); 
  this->swap(result);
}

#endif // #ifndef SILL_TABLE_AND_LAZY_FACTOR_HPP
