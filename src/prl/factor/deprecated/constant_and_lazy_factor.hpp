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

#ifndef PRL_CONSTANT_AND_LAZY_FACTOR_HPP
#define PRL_CONSTANT_AND_LAZY_FACTOR_HPP

//! \todo put the #ifdefs over here

#include <prl/factor/constant_factor.hpp>
#include <prl/factor/lazy_factor.hpp>

template <typename range_t>
template <typename internal_factor_t,
          typename csr_tag_t,
          typename elim_strategy_t,
          typename binary_op_tag_t>
prl::constant_factor<range_t>::constant_factor
(factor_collapse_expr<lazy_factor_t<internal_factor_t,
                                      csr_tag_t,
                                      elim_strategy_t>,
                        binary_op_tag_t> expr) {
  // This conversion is possible only if all variables are collapsed
  // out.
  assert(expr.retained.empty());
  // First collapse in the lazy factor representation.
  lazy_factor_t<internal_factor_t, csr_tag_t, elim_strategy_t>
    collapsed(prl::collapse(expr.x_ptr, expr.retained, binary_op_tag_t()));
  // To get the constant value of this factor (which has no
  // arguments), convert it to a constant factor.
  constant_factor<range_t>
    constant(prl::collapse(collapsed.flatten(),
                           domain::empty_set,
                           binary_op_tag_t()));
  this->value = constant.get_value();
}

#endif // #ifndef PRL_CONSTANT_AND_LAZY_FACTOR_HPP



