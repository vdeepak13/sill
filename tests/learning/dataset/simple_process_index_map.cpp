#define BOOST_TEST_MODULE simple_process_index_map
#include <boost/test/unit_test.hpp>

#include <sill/learning/dataset/simple_process_index_map.hpp>

using namespace sill;

template class simple_process_index_map<finite_variable>;

typedef std::pair<size_t,size_t> index_pair;
BOOST_TEST_DONT_PRINT_LOG_VALUE(index_pair);


BOOST_AUTO_TEST_CASE(test_all) {
  std::vector<finite_discrete_process*> procs;
  procs.push_back(new finite_discrete_process("a", 2));
  procs.push_back(new finite_discrete_process("b", 3));
  procs.push_back(new finite_discrete_process("c", 4));
  
  simple_process_index_map<finite_variable> map;
  map.initialize(procs);
  
  std::vector<finite_discrete_process*> procs2;
  procs2.push_back(procs[2]);
  procs2.push_back(procs[1]);
  std::vector<size_t> proc_indices;
  map.indices(procs2, proc_indices);
  
  BOOST_CHECK_EQUAL(proc_indices.size(), 2);
  BOOST_CHECK_EQUAL(proc_indices[0], 2);
  BOOST_CHECK_EQUAL(proc_indices[1], 1);

  std::vector<finite_variable*> vars;
  vars.push_back(procs[2]->at(3));
  vars.push_back(procs[0]->at(1));
  vars.push_back(procs[1]->current());
  vars.push_back(procs[1]->next());
  std::vector<std::pair<size_t,size_t> > var_indices;
  map.indices(vars, 2, var_indices);
  BOOST_CHECK_EQUAL(var_indices.size(), 4);
  BOOST_CHECK_EQUAL(var_indices[0], index_pair(2, 5));
  BOOST_CHECK_EQUAL(var_indices[1], index_pair(0, 3));
  BOOST_CHECK_EQUAL(var_indices[2], index_pair(1, 2));
  BOOST_CHECK_EQUAL(var_indices[3], index_pair(1, 3));
}
