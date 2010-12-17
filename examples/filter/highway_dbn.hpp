#pragma once

#include <vector>

#include <prl/factor/table_factor.hpp>
#include <prl/model/dynamic_bayesian_network.hpp>
#include <prl/base/timed_process.hpp>

/**
 * Makes a DBN modeling traffic speed on a highway divided into a
 * fixed number of segments.  
 * \param n The number of segments.
 * \param dbn The output DBN.
 * \param procs The state process for each segment.
 **/
void highway_dbn(std::size_t n, 
                 prl::dynamic_bayesian_network<prl::table_factor>& dbn,
                 std::vector<prl::finite_timed_process*>& procs);
