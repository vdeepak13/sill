#define BOOST_TEST_MODULE belief_update_calibrate
#include <boost/test/unit_test.hpp>

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/probability_table.hpp>
#include <sill/inference/exact/belief_update_calibrate.hpp>

namespace sill {
  template class belief_update_calibrate<ptable>;
  template class belief_update_calibrate<cgaussian>;
}

#include "mn_fixture.hpp"

BOOST_FIXTURE_TEST_CASE(test_calibrate, fixture) {
  belief_update_calibrate<ptable> engine(mn);

  // check if clique marginals are correct
  engine.calibrate();
  for (size_t v : engine.jt().vertices()) {
    check_belief(engine.belief(v), 1e-8);
  }

  // check if clique marginals are correct after normalization
  engine.normalize();
  for (size_t v : engine.jt().vertices()) {
    check_belief_normalized(engine.belief(v), 1e-8);
  }
}