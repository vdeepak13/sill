#define BOOST_TEST_MODULE vector_sequence_record
#include <boost/test/unit_test.hpp>

#include <sill/learning/dataset/vector_sequence_record.hpp>

using namespace sill;

typedef std::vector<vector_discrete_process*> proc_vector_type;

template class vector_sequence_record<double>;

struct fixture {
  fixture() {
    vector_discrete_process* x = new vector_discrete_process("x", 2);
    vector_discrete_process* y = new vector_discrete_process("y", 1);
    procs.push_back(x);
    procs.push_back(y);
  }
  proc_vector_type procs;
};

BOOST_FIXTURE_TEST_CASE(test_construct, fixture) {
  vector_sequence_record<> r(procs);
  BOOST_CHECK_EQUAL(r.num_processes(), 2);
  BOOST_CHECK(std::equal(procs.begin(), procs.end(), r.processes()));
  BOOST_CHECK_EQUAL(r.num_dims(), 3);
  BOOST_CHECK_EQUAL(r.num_steps(), 0);
  BOOST_CHECK_EQUAL(r.size(), 0);
  BOOST_CHECK_EQUAL(r.weight(), 1.0);
  BOOST_CHECK_THROW(r.check_compatible(proc_vector_type()), std::invalid_argument);
  r.check_compatible(procs);
  vector_assignment a;
  r.extract(a);
  BOOST_CHECK_EQUAL(a.size(), 0);
}


BOOST_FIXTURE_TEST_CASE(test_assign, fixture) {
  vector_sequence_record<> r(procs);

  // test vector assignment
  std::vector<double> valvec;
  valvec.push_back(1);
  valvec.push_back(2);
  valvec.push_back(4);
  valvec.push_back(8);
  valvec.push_back(16);
  valvec.push_back(32);
  r.assign(valvec, 0.5);
  BOOST_CHECK_EQUAL(r.num_steps(), 2);
  BOOST_CHECK_EQUAL(r.size(), 6);
  BOOST_CHECK_EQUAL(r.weight(), 0.5);
  BOOST_CHECK_EQUAL(r.value_ptr(0, 0)[0], 1);
  BOOST_CHECK_EQUAL(r.value_ptr(0, 1)[0], 2);
  BOOST_CHECK_EQUAL(r.value_ptr(0, 0)[2], 4);
  BOOST_CHECK_EQUAL(r.value_ptr(0, 1)[2], 8);
  BOOST_CHECK_EQUAL(r.value_ptr(1, 0)[0], 16);
  BOOST_CHECK_EQUAL(r.value_ptr(1, 1)[0], 32);
  std::cout << r << std::endl;
  r.free_memory();

  // test pointer assignment
  double* values = new double[3];
  values[0] = 1;
  values[1] = 0.5;
  values[2] = 0.25;
  r.assign(values, 1, 1.0);
  BOOST_CHECK_EQUAL(r.num_steps(), 1);
  BOOST_CHECK_EQUAL(r.size(), 3);
  BOOST_CHECK_EQUAL(r.weight(), 1.0);
  BOOST_CHECK_EQUAL(r.value_ptr(0, 0)[0], 1);
  BOOST_CHECK_EQUAL(r.value_ptr(0, 0)[1], 0.5);
  BOOST_CHECK_EQUAL(r.value_ptr(1, 0)[0], 0.25);
  r.free_memory();

  // test vector_assignment assignment
  vector_assignment a;
  a[procs[0]->at(0)] = "1 2";
  a[procs[1]->at(0)] = "4";
  a[procs[0]->at(1)] = "8 16";
  a[procs[1]->at(1)] = "32";
  r.assign(a, 2.0);
  BOOST_CHECK_EQUAL(r.num_steps(), 2);
  BOOST_CHECK_EQUAL(r.size(), 6);
  BOOST_CHECK_EQUAL(r.weight(), 2.0);
  BOOST_CHECK_EQUAL(r.value_ptr(0, 0)[0], 1);
  BOOST_CHECK_EQUAL(r.value_ptr(0, 0)[2], 2);
  BOOST_CHECK_EQUAL(r.value_ptr(1, 0)[0], 4);
  BOOST_CHECK_EQUAL(r.value_ptr(0, 1)[0], 8);
  BOOST_CHECK_EQUAL(r.value_ptr(0, 1)[2], 16);
  BOOST_CHECK_EQUAL(r.value_ptr(1, 1)[0], 32);
  r.free_memory();
}

BOOST_FIXTURE_TEST_CASE(test_extract, fixture) {
  vector_sequence_record<> r(procs);
  double* values = new double[6];
  values[0] = 1;
  values[1] = 2;
  values[2] = 4;
  values[3] = 8;
  values[4] = 16;
  values[5] = 32;
  r.assign(values, 2, 0.5);

  // test extract all
  vector_assignment a_all;
  r.extract(a_all);
  BOOST_CHECK_EQUAL(a_all.size(), 4);
  BOOST_CHECK(equal(a_all[procs[0]->at(0)], vec("1 4")));
  BOOST_CHECK(equal(a_all[procs[0]->at(1)], vec("2 8")));
  BOOST_CHECK(equal(a_all[procs[1]->at(0)], vec("16")));
  BOOST_CHECK(equal(a_all[procs[1]->at(1)], vec("32")));
  
  // test extract current
  vector_assignment a_cur;
  r.extract(1, a_cur);
  BOOST_CHECK_EQUAL(a_cur.size(), 2);
  BOOST_CHECK(equal(a_cur[procs[0]->current()], vec("2 8")));
  BOOST_CHECK(equal(a_cur[procs[1]->current()], vec("32")));
  
  // test extract pointers
  std::vector<std::pair<size_t,size_t> > indices;
  indices.push_back(std::make_pair(1, 0));
  indices.push_back(std::make_pair(0, 1));
  raw_record_iterator_state<vector_record<> > state;
  r.extract(indices, state);
  BOOST_CHECK_EQUAL(state.elems.size(), 3);
  BOOST_CHECK_EQUAL(state.elems[0], values+4);
  BOOST_CHECK_EQUAL(state.elems[1], values+1);
  BOOST_CHECK_EQUAL(state.elems[2], values+3);
  BOOST_CHECK_EQUAL(*state.weights, 0.5);
  
  // test extract record
  vector_record<> rt;
  r.extract(indices, rt);
  BOOST_CHECK(equal(rt.values, vec("16 2 8")));
  BOOST_CHECK_EQUAL(rt.weight, 0.5);
}

BOOST_FIXTURE_TEST_CASE(test_load, fixture) {
  vector_sequence_record<> r(procs);
  double* values = new double[6];
  r.assign(values, 2, 0.5);

  vector_sequence_record<> r2(proc_vector_type(1, procs[1]), 2.0);
  std::vector<size_t> indices;
  indices.push_back(1);
  r2.load(r, indices);
  
  BOOST_CHECK_EQUAL(r2.num_processes(), 1);
  BOOST_CHECK_EQUAL(r2.num_steps(), 2);
  BOOST_CHECK_EQUAL(r2.weight(), 0.5);
  BOOST_CHECK_EQUAL(r2.value_ptr(0, 1), values + 5);
}
