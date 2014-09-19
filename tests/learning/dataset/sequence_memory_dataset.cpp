#define BOOST_TEST_MODULE sequence_memory_dataset
#include <boost/test/unit_test.hpp>

#include <sill/learning/dataset/finite_dataset.hpp>
#include <sill/learning/dataset/finite_sequence_record.hpp>
#include <sill/learning/dataset/hybrid_dataset.hpp>
#include <sill/learning/dataset/hybrid_sequence_record.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/dataset/vector_sequence_record.hpp>
#include <sill/learning/dataset/sequence_memory_dataset.hpp>
#include <sill/learning/dataset/slice_view.hpp>

#include <boost/random/mersenne_twister.hpp>

using namespace sill;

template class sequence_dataset<finite_dataset>;
template class sequence_dataset<vector_dataset<> >;
template class sequence_dataset<hybrid_dataset<> >;

template class sequence_memory_dataset<finite_dataset>;
template class sequence_memory_dataset<vector_dataset<> >;
template class sequence_memory_dataset<hybrid_dataset<> >;

template class slice_view<sequence_dataset<finite_dataset> >;

BOOST_AUTO_TEST_CASE(test_insert) {
  std::vector<finite_discrete_process*> procs(3);
  procs[0] = new finite_discrete_process("a", 2);
  procs[1] = new finite_discrete_process("b", 3);
  procs[2] = new finite_discrete_process("c", 4);
  
  sequence_memory_dataset<finite_dataset> ds;
  ds.initialize(procs);
  
  // insert a record
  finite_sequence_record r(procs);
  size_t* values = new size_t[6];
  values[0] = 0;
  values[1] = 1;
  values[2] = 1;
  values[3] = 2;
  values[4] = 2;
  values[5] = 3;
  r.assign(values, 2, 0.5);
  ds.insert(r);
  
  // insert another record
  std::vector<size_t> values2(3);
  values2[0] = 1;
  values2[1] = 2;
  values2[2] = 3;
  r.assign(values2, 1.0);
  ds.insert(r);
  
  // insert an assignment
  finite_assignment a;
  a[procs[0]->at(0)] = 1;
  a[procs[0]->at(1)] = 0;
  a[procs[1]->at(0)] = 1;
  a[procs[1]->at(1)] = 2;
  a[procs[2]->at(0)] = 3;
  a[procs[2]->at(1)] = 2;
  ds.insert(a, 2.0);
  
  // print the datset
  std::cout << ds << std::endl;
  
  // basic checks
  BOOST_CHECK_EQUAL(ds.size(), 3);
  BOOST_CHECK_EQUAL(ds.empty(), false);

  sequence_dataset<finite_dataset>::const_record_iterator it, end;
  boost::tie(it, end) = ds.records(procs);

  // check the first record
  finite_sequence_record r0 = *it;
  BOOST_CHECK_EQUAL(r0.num_processes(), 3);
  BOOST_CHECK_EQUAL(r0.num_steps(), 2);
  BOOST_CHECK_EQUAL(r0.weight(), 0.5);
  BOOST_CHECK_EQUAL(r0(0, 0), 0);
  BOOST_CHECK_EQUAL(r0(0, 1), 1);
  BOOST_CHECK_EQUAL(r0(1, 0), 1);
  BOOST_CHECK_EQUAL(r0(1, 1), 2);
  BOOST_CHECK_EQUAL(r0(2, 0), 2);
  BOOST_CHECK_EQUAL(r0(2, 1), 3);
  ++it;

  // check the second record
  finite_sequence_record r1 = *it;
  BOOST_CHECK_EQUAL(r1.num_processes(), 3);
  BOOST_CHECK_EQUAL(r1.num_steps(), 1);
  BOOST_CHECK_EQUAL(r1.weight(), 1.0);
  BOOST_CHECK_EQUAL(r1(0, 0), 1);
  BOOST_CHECK_EQUAL(r1(1, 0), 2);
  BOOST_CHECK_EQUAL(r1(2, 0), 3);
  ++it;

  // check the third record
  finite_sequence_record r2 = *it;
  BOOST_CHECK_EQUAL(r2.num_processes(), 3);
  BOOST_CHECK_EQUAL(r2.num_steps(), 2);
  BOOST_CHECK_EQUAL(r2.weight(), 2.0);
  BOOST_CHECK_EQUAL(r2(0, 0), 1);
  BOOST_CHECK_EQUAL(r2(0, 1), 0);
  BOOST_CHECK_EQUAL(r2(1, 0), 1);
  BOOST_CHECK_EQUAL(r2(1, 1), 2);
  BOOST_CHECK_EQUAL(r2(2, 0), 3);
  BOOST_CHECK_EQUAL(r2(2, 1), 2);
  ++it;

  BOOST_CHECK(it == end);

  // check a single record extraction
  std::vector<finite_discrete_process*> procs21;
  procs21.push_back(procs[2]);
  procs21.push_back(procs[1]);
  finite_sequence_record ri = ds.record(2, procs21);
  BOOST_CHECK_EQUAL(ri.num_processes(), 2);
  BOOST_CHECK_EQUAL(ri.num_steps(), 2);
  BOOST_CHECK_EQUAL(ri.weight(), 2.0);
  BOOST_CHECK_EQUAL(ri(0, 0), 3);
  BOOST_CHECK_EQUAL(ri(0, 1), 2);
  BOOST_CHECK_EQUAL(ri(1, 0), 1);
  BOOST_CHECK_EQUAL(ri(1, 1), 2);

  // instantiate the sample() template
  boost::mt19937 rng;
  ds.sample(procs21, rng);
  
  // instantiate the shuffle template
  ds.shuffle(rng);
}

BOOST_AUTO_TEST_CASE(test_iterator_comparisons) {
  std::vector<finite_discrete_process*> procs(3);
  procs[0] = new finite_discrete_process("a", 2);
  procs[1] = new finite_discrete_process("b", 3);
  procs[2] = new finite_discrete_process("c", 4);
  
  sequence_memory_dataset<finite_dataset> ds;
  ds.initialize(procs);
  ds.insert(finite_sequence_record(procs));

  sequence_dataset<finite_dataset>::record_iterator it1, end1;
  boost::tie(it1, end1) = ds.records(procs);

  sequence_dataset<finite_dataset>::const_record_iterator it2, end2;
  boost::tie(it2, end2) = ds.records(procs);

  BOOST_CHECK(it1 == it2);
  BOOST_CHECK(it2 == it1);

  BOOST_CHECK(end1 == end2);
  BOOST_CHECK(end2 == end1);

  BOOST_CHECK(it1 != end1);
  BOOST_CHECK(it1 != end2);
  BOOST_CHECK(it2 != end1);
  BOOST_CHECK(it2 != end2);

  BOOST_CHECK(++it1 == end1);
  BOOST_CHECK(++it2 == end2);
}
