#pragma once

#include <vector>

#include <sill/factor/table_factor.hpp>
#include <sill/model/dynamic_bayesian_network.hpp>
#include <sill/base/discrete_process.hpp>

/**
 * Makes a DBN modeling traffic speed on a highway divided into a
 * fixed number of segments.  
 * \param n The number of segments.
 * \param dbn The output DBN.
 * \param procs The state process for each segment.
 **/
void highway_dbn(std::size_t n, 
                 sill::dynamic_bayesian_network<sill::table_factor>& dbn,
                 std::vector<sill::finite_discrete_process*>& procs);
