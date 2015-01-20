#define BOOST_TEST_MODULE sequence_dataset_views
#include <boost/test/unit_test.hpp>

#include <sill/learning/dataset/finite_dataset.hpp>
#include <sill/learning/dataset/finite_sequence_record.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/dataset/vector_sequence_record.hpp>
#include <sill/learning/dataset/hybrid_dataset.hpp>
#include <sill/learning/dataset/hybrid_sequence_record.hpp>
#include <sill/learning/dataset/sequence_memory_dataset.hpp>

namespace sill {
  template class sliding_view<finite_dataset>;
  template class sliding_view<vector_dataset<> >;
  template class sliding_view<hybrid_dataset<> >;
  
  template class fixed_view<finite_dataset>;
  template class fixed_view<vector_dataset<> >;
  template class fixed_view<hybrid_dataset<> >;
}

using namespace sill;

struct fixture {
  fixture() {
    procs.resize(3);
    procs[0] = new finite_discrete_process("a", 2);
    procs[1] = new finite_discrete_process("b", 3);
    procs[2] = new finite_discrete_process("c", 4);
  
    ds.initialize(procs);
  
    // insert a record
    finite_sequence_record r(procs);
    size_t* values1 = new size_t[6];
    values1[0] = 0;
    values1[1] = 1;
    values1[2] = 1;
    values1[3] = 2;
    values1[4] = 2;
    values1[5] = 3;
    r.assign(values1, 2, 0.5);
    ds.insert(r);

    // insert another record
    size_t* values2 = new size_t[3];
    values2[0] = 1;
    values2[1] = 2;
    values2[2] = 0;
    r.assign(values2, 1, 2.0);
    ds.insert(r);
  }

  std::vector<finite_discrete_process*> procs;
  sequence_memory_dataset<finite_dataset> ds;
};

BOOST_FIXTURE_TEST_CASE(sliding_window0, fixture) {
  sliding_view<finite_dataset> view = ds.sliding(0);
  finite_dataset::const_record_iterator it, end;
  finite_var_vector vars = variables(procs, current_step);
  boost::tie(it, end) = view.records(vars);

  // check the arguments
  BOOST_CHECK_EQUAL(it->variables, vars);

  // check the first record
  BOOST_CHECK_EQUAL(it->values.size(), 3);
  BOOST_CHECK_EQUAL(it->values[0], 0);
  BOOST_CHECK_EQUAL(it->values[1], 1);
  BOOST_CHECK_EQUAL(it->values[2], 2);
  BOOST_CHECK_EQUAL(it->weight, 0.5);
  ++it;

  // check the second record
  BOOST_CHECK_EQUAL(it->values.size(), 3);
  BOOST_CHECK_EQUAL(it->values[0], 1);
  BOOST_CHECK_EQUAL(it->values[1], 2);
  BOOST_CHECK_EQUAL(it->values[2], 3);
  BOOST_CHECK_EQUAL(it->weight, 0.5);
  ++it;

  // check the third record
  BOOST_CHECK_EQUAL(it->values.size(), 3);
  BOOST_CHECK_EQUAL(it->values[0], 1);
  BOOST_CHECK_EQUAL(it->values[1], 2);
  BOOST_CHECK_EQUAL(it->values[2], 0);
  BOOST_CHECK_EQUAL(it->weight, 2.0);
  ++it;

  // have we reached the end?
  BOOST_CHECK(it == end);
}

BOOST_FIXTURE_TEST_CASE(sliding_window1, fixture) {
  sliding_view<finite_dataset> view = ds.sliding(1);
  finite_dataset::const_record_iterator it, end;
  finite_var_vector vars0 = variables(procs, current_step);
  finite_var_vector vars1 = variables(procs, next_step);
  finite_var_vector vars = concat(vars0, vars1);
  boost::tie(it, end) = view.records(vars);

  // check the arguments
  BOOST_CHECK_EQUAL(it->variables, vars);

  // check the record
  BOOST_CHECK_EQUAL(it->values.size(), 6);
  BOOST_CHECK_EQUAL(it->values[0], 0);
  BOOST_CHECK_EQUAL(it->values[1], 1);
  BOOST_CHECK_EQUAL(it->values[2], 2);
  BOOST_CHECK_EQUAL(it->values[3], 1);
  BOOST_CHECK_EQUAL(it->values[4], 2);
  BOOST_CHECK_EQUAL(it->values[5], 3);
  BOOST_CHECK_EQUAL(it->weight, 0.5);
  ++it;
  
  // have we reached the end?
  BOOST_CHECK(it == end);
}

BOOST_FIXTURE_TEST_CASE(sliding_window_selected, fixture) {
  sliding_view <finite_dataset> view = ds.sliding(1);
  finite_dataset::const_record_iterator it, end;
  finite_var_vector vars;
  vars.push_back(procs[0]->current());
  vars.push_back(procs[2]->current());
  vars.push_back(procs[1]->next());
  boost::tie(it, end) = view.records(vars);

  // check the arguments
  BOOST_CHECK_EQUAL(it->variables, vars);

  // check the record
  BOOST_CHECK_EQUAL(it->values.size(), 3);
  BOOST_CHECK_EQUAL(it->values[0], 0);
  BOOST_CHECK_EQUAL(it->values[1], 2);
  BOOST_CHECK_EQUAL(it->values[2], 2);
  BOOST_CHECK_EQUAL(it->weight, 0.5);
  ++it;

  // have we reached the end?
  BOOST_CHECK(it == end);
}

BOOST_FIXTURE_TEST_CASE(sliding_window_record, fixture) {
  sliding_view<finite_dataset> view = ds.sliding(0);
  finite_var_vector vars = variables(procs, current_step);
  finite_record r = view.record(2, vars);

  BOOST_CHECK_EQUAL(r.variables, vars);
  BOOST_CHECK_EQUAL(r.values.size(), 3);
  BOOST_CHECK_EQUAL(r.values[0], 1);
  BOOST_CHECK_EQUAL(r.values[1], 2);
  BOOST_CHECK_EQUAL(r.values[2], 0);
  BOOST_CHECK_EQUAL(r.weight, 2.0);
}

BOOST_FIXTURE_TEST_CASE(fixed0, fixture) {
  fixed_view<finite_dataset> view = ds.fixed(0);
  finite_dataset::const_record_iterator it, end;
  finite_var_vector vars = variables(procs, current_step);
  boost::tie(it, end) = view.records(vars);
  
  // check the arguments
  BOOST_CHECK_EQUAL(it->variables, vars);

  // check the first record
  BOOST_CHECK_EQUAL(it->values.size(), 3);
  BOOST_CHECK_EQUAL(it->values[0], 0);
  BOOST_CHECK_EQUAL(it->values[1], 1);
  BOOST_CHECK_EQUAL(it->values[2], 2);
  BOOST_CHECK_EQUAL(it->weight, 0.5);
  ++it;

  // check the third record
  BOOST_CHECK_EQUAL(it->values.size(), 3);
  BOOST_CHECK_EQUAL(it->values[0], 1);
  BOOST_CHECK_EQUAL(it->values[1], 2);
  BOOST_CHECK_EQUAL(it->values[2], 0);
  BOOST_CHECK_EQUAL(it->weight, 2.0);
  ++it;

  // have we reached the end?
  BOOST_CHECK(it == end);
}

/*
BOOST_FIXTURE_TEST_CASE(fixed01_selected, fixture) {
  sliding_view <finite_dataset> view = ds.sliding(1);
  finite_dataset::const_record_iterator it, end;
  finite_var_vector vars;
  vars.push_back(procs[0]->at(1));
  vars.push_back(procs[2]->at(1));
  vars.push_back(procs[1]->at(0));
  boost::tie(it, end) = view.records(vars);

  // check the arguments
  BOOST_CHECK_EQUAL(it->variables, vars);

  // check the record
  BOOST_CHECK_EQUAL(it->values.size(), 3);
  BOOST_CHECK_EQUAL(it->values[0], 1);
  BOOST_CHECK_EQUAL(it->values[1], 3);
  BOOST_CHECK_EQUAL(it->values[2], 1);
  BOOST_CHECK_EQUAL(it->weight, 0.5);
  ++it;

  // have we reached the end?
  BOOST_CHECK(it == end);
}

BOOST_FIXTURE_TEST_CASE(fixed_record_selected, fixture) {
  fixed_view<finite_dataset> view = ds.fixed(0);
  finite_var_vector vars;
  vars.push_back(procs[0]->at(1));
  vars.push_back(procs[2]->at(0));
  finite_record r = view.record(0, vars);

  BOOST_CHECK_EQUAL(r.variables, vars);
  BOOST_CHECK_EQUAL(r.values.size(), 2);
  BOOST_CHECK_EQUAL(r.values[0], 1);
  BOOST_CHECK_EQUAL(r.values[1], 2);
  BOOST_CHECK_EQUAL(r.weight, 0.5);
}
*/
