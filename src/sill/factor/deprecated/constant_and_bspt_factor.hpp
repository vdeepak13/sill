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

#ifndef SILL_CONSTANT_AND_BSPT_FACTOR_HPP
#define SILL_CONSTANT_AND_BSPT_FACTOR_HPP

//! \todo put the #ifdefs over here

#include <sill/factor/constant_factor.hpp>
#include <sill/factor/bspt_factor.hpp>

template <typename range_t>
template <typename binary_op_tag_t,
          typename leaf_factor_t>
sill::constant_factor<range_t>::constant_factor
(factor_collapse_expr<bspt_factor_t<leaf_factor_t>,
                        binary_op_tag_t> expr) {
  // This conversion is possible only if all variables are collapsed
  // out.
  assert(expr.x_ptr->arguments().intersection_size(expr.retained) == 0);
  // Collapse the BSPT factor over all its arguments.
  bspt_factor_t<leaf_factor_t> f(sill::collapse(expr.x_ptr,
                                               domain::empty_set,
                                               sum_tag()));
  // Get the only internal factor in the tree.
  assert(f.is_singleton());
  const_ptr_t<leaf_factor_t> g_ptr = f.get_singleton_factor();
  // Convert it to a constant factor using a collapse operation.  All
  // of the arguments have already been collapsed out, but this is a
  // simple way to exploit existing conversions from internal factors
  // to constant factors.
  constant_factor<range_t> h(sill::collapse(g_ptr, domain::empty_set,
                                             sum_tag()));
  this->value = h.get_value();
}

#endif // #ifndef SILL_CONSTANT_AND_BSPT_FACTOR_HPP



