#define BOOST_TEST_MODULE finite_sequence_record
#include <boost/test/unit_test.hpp>

#include <sill/learning/dataset/finite_sequence_record.hpp>

using namespace sill;

typedef std::vector<finite_discrete_process*> proc_vector_type;

struct fixture {
  fixture() {
    finite_discrete_process* a = new finite_discrete_process("a", 2);
    finite_discrete_process* b = new finite_discrete_process("b", 4);
    procs.push_back(a);
    procs.push_back(b);
  }
  proc_vector_type procs;
};

//BOOST_TEST_DONT_PRINT_LOG_VALUE(std::vector<finite_discrete_process*>);

BOOST_FIXTURE_TEST_CASE(test_construct, fixture) {
  finite_sequence_record r(procs);
  BOOST_CHECK_EQUAL(r.num_processes(), 2);
  BOOST_CHECK(std::equal(procs.begin(), procs.end(), r.processes()));
  BOOST_CHECK_EQUAL(r.num_steps(), 0);
  BOOST_CHECK_EQUAL(r.size(), 0);
  BOOST_CHECK_EQUAL(r.weight(), 1.0);
  BOOST_CHECK_THROW(r.check_compatible(proc_vector_type()), std::invalid_argument);
  r.check_compatible(procs);
  finite_assignment a;
  r.extract(a);
  BOOST_CHECK_EQUAL(a.size(), 0);
}


BOOST_FIXTURE_TEST_CASE(test_assign, fixture) {
  finite_sequence_record r(procs);

  // vector assignment
  std::vector<size_t> valvec;
  valvec.push_back(0);
  valvec.push_back(1);
  valvec.push_back(0);
  valvec.push_back(1);
  valvec.push_back(2);
  valvec.push_back(0);
  r.assign(valvec, 0.5);
  BOOST_CHECK_EQUAL(r.num_steps(), 3);
  BOOST_CHECK_EQUAL(r.size(), 6);
  BOOST_CHECK_EQUAL(r.weight(), 0.5);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t t = 0; t < 3; ++t) {
      BOOST_CHECK_EQUAL(r(i, t), valvec[i*3 + t]);
    }
  }
  std::cout << r << std::endl;
  r.free_memory();

  // pointer assignment
  size_t* values = new size_t[4];
  values[0] = 1;
  values[1] = 0;
  values[2] = 2;
  values[3] = 1;
  r.assign(values, 2, 1.0);
  BOOST_CHECK_EQUAL(r.num_steps(), 2);
  BOOST_CHECK_EQUAL(r.size(), 4);
  BOOST_CHECK_EQUAL(r.weight(), 1.0);
  BOOST_CHECK_EQUAL(r(0, 0), 1);
  BOOST_CHECK_EQUAL(r(0, 1), 0);
  BOOST_CHECK_EQUAL(r(1, 0), 2);
  BOOST_CHECK_EQUAL(r(1, 1), 1);
  r.free_memory();

  // finite_assignment assignment
  finite_assignment a;
  a[procs[0]->at(1)] = 1;
  a[procs[0]->at(0)] = 0;
  a[procs[1]->at(1)] = 2;
  a[procs[1]->at(0)] = 1;
  r.assign(a, 2.0);
  BOOST_CHECK_EQUAL(r.num_steps(), 2);
  BOOST_CHECK_EQUAL(r.size(), 4);
  BOOST_CHECK_EQUAL(r.weight(), 2.0);
  BOOST_CHECK_EQUAL(r(0, 0), 0);
  BOOST_CHECK_EQUAL(r(0, 1), 1);
  BOOST_CHECK_EQUAL(r(1, 0), 1);
  BOOST_CHECK_EQUAL(r(1, 1), 2);

  // setting a particular time step
  std::vector<size_t> values1;
  values1.push_back(2);
  values1.push_back(1);
  r.set(1, values1);
  BOOST_CHECK_EQUAL(r.num_steps(), 2);
  BOOST_CHECK_EQUAL(r.size(), 4);
  BOOST_CHECK_EQUAL(r.weight(), 2.0);
  BOOST_CHECK_EQUAL(r(0, 0), 0);
  BOOST_CHECK_EQUAL(r(0, 1), 2);
  BOOST_CHECK_EQUAL(r(1, 0), 1);
  BOOST_CHECK_EQUAL(r(1, 1), 1);

  r.free_memory();
}

BOOST_FIXTURE_TEST_CASE(test_extract, fixture) {
  finite_sequence_record r(procs);
  size_t* values = new size_t[6];
  values[0] = 0;
  values[1] = 1;
  values[2] = 0;
  values[3] = 1;
  values[4] = 2;
  values[5] = 3;
  r.assign(values, 3, 0.5);

  // extract an assignment for all
  finite_assignment a_all;
  r.extract(a_all);
  BOOST_CHECK_EQUAL(a_all.size(), 6);
  BOOST_CHECK_EQUAL(a_all[procs[0]->at(0)], 0);
  BOOST_CHECK_EQUAL(a_all[procs[0]->at(1)], 1);
  BOOST_CHECK_EQUAL(a_all[procs[0]->at(2)], 0);
  BOOST_CHECK_EQUAL(a_all[procs[1]->at(0)], 1);
  BOOST_CHECK_EQUAL(a_all[procs[1]->at(1)], 2);
  BOOST_CHECK_EQUAL(a_all[procs[1]->at(2)], 3);
  
  // extract assignment at t
  finite_assignment a_cur;
  r.extract(1, a_cur);
  BOOST_CHECK_EQUAL(a_cur.size(), 2);
  BOOST_CHECK_EQUAL(a_cur[procs[0]->current()], 1);
  BOOST_CHECK_EQUAL(a_cur[procs[1]->current()], 2);

  // extract values at t
  std::vector<size_t> values1;
  r.extract(1, values1);
  BOOST_CHECK_EQUAL(values1.size(), 2);
  BOOST_CHECK_EQUAL(values1[0], 1);
  BOOST_CHECK_EQUAL(values1[1], 2);
  
  // extract values for a time range
  std::vector<size_t> values12;
  r.extract(1, 2, values12);
  BOOST_CHECK_EQUAL(values12.size(), 4);
  BOOST_CHECK_EQUAL(values12[0], 1);
  BOOST_CHECK_EQUAL(values12[1], 2);
  BOOST_CHECK_EQUAL(values12[2], 0);
  BOOST_CHECK_EQUAL(values12[3], 3);
  
  // extract pointers
  std::vector<size_t*> ptrs;
  std::vector<std::pair<size_t,size_t> > indices;
  indices.push_back(std::make_pair(1, 2));
  indices.push_back(std::make_pair(0, 0));
  raw_record_iterator_state<finite_record> state;
  r.extract(indices, state);
  BOOST_CHECK_EQUAL(state.elems.size(), 2);
  BOOST_CHECK_EQUAL(state.elems[0], values+5);
  BOOST_CHECK_EQUAL(state.elems[1], values+0);
  BOOST_CHECK_EQUAL(*state.weights, 0.5);
  
  // extract record
  finite_record rt;
  r.extract(indices, rt);
  BOOST_CHECK_EQUAL(rt.values.size(), 2);
  BOOST_CHECK_EQUAL(rt.values[0], 3);
  BOOST_CHECK_EQUAL(rt.values[1], 0);
  BOOST_CHECK_EQUAL(rt.weight, 0.5);
}

BOOST_FIXTURE_TEST_CASE(test_load, fixture) {
  finite_sequence_record r(procs);
  size_t* values = new size_t[6];
  r.assign(values, 3, 0.5);

  finite_sequence_record r2(proc_vector_type(1, procs[1]), 2.0);
  std::vector<size_t> indices;
  indices.push_back(1);
  r2.load(r, indices);
  
  BOOST_CHECK_EQUAL(r2.num_processes(), 1);
  BOOST_CHECK_EQUAL(r2.num_steps(), 3);
  BOOST_CHECK_EQUAL(r2.weight(), 0.5);
  BOOST_CHECK_EQUAL(&r2(0, 1), values + 4);
}
