#define BOOST_TEST_MODULE learnt_decomposable
#include <boost/test/unit_test.hpp>

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/learnt_decomposable.hpp>

#include "basic_fixture.hpp"

template class learnt_decomposable<canonical_gaussian>;
template class learnt_decomposable<table_factor>;

BOOST_FIXTURE_TEST_CASE(test_mutation, basic_fixture) {
  typedef learnt_decomposable<table_factor>::vertex vertex;

  table_factor cvp_factor(make_domain(lvedvolume, cvp), 1.0);
  finite_domain lvedvolume_domain = make_domain(lvedvolume);

  learnt_decomposable<table_factor> model;
  model *= factors;

  vertex cvp_connect = model.find_clique_cover(lvedvolume_domain);
  vertex cvp_vertex = model.add_clique(cvp_factor.arguments(), cvp_factor);
  model.add_edge(cvp_vertex, cvp_connect);
  
  model.check_validity();
}
