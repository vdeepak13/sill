#define BOOST_TEST_MODULE hybrid_sequence_record
#include <boost/test/unit_test.hpp>

#include <sill/learning/dataset/hybrid_sequence_record.hpp>

namespace sill {
  template class hybrid_sequence_record<double>;
  template void load_tabular(const std::string&, const symbolic_format&,
                             hybrid_sequence_record<>&);
  template void save_tabular(const std::string&, const symbolic_format&,
                             const hybrid_sequence_record<>&);
}

using namespace sill;

typedef std::vector<discrete_process<variable>*> proc_vector_type;
typedef discrete_process<variable> process_type;

struct fixture {
  fixture() {
    finite_discrete_process* x = new finite_discrete_process("x", 3);
    vector_discrete_process* y = new vector_discrete_process("y", 2);
    procs.push_back(new process_type(x));
    procs.push_back(new process_type(y));
  }
  proc_vector_type procs;
};

BOOST_FIXTURE_TEST_CASE(test_construct, fixture) {
  hybrid_sequence_record<> r(procs, 2.0);
  BOOST_CHECK_EQUAL(r.num_processes(), 2);
  BOOST_CHECK_EQUAL(r.num_steps(), 0);
  BOOST_CHECK_EQUAL(r.size(), 0);
  BOOST_CHECK_EQUAL(r.weight(), 2.0);
  r.check_compatible(procs);

  assignment a;
  r.extract(a);
  BOOST_CHECK_EQUAL(a.size(), 0);
}

BOOST_FIXTURE_TEST_CASE(test_operations, fixture) {
  // test assignment
  hybrid_sequence_record<> r(procs, 2.0);
  std::vector<size_t> finite_vals(2, 1);
  std::vector<double> vector_vals(4, 2);
  r.assign(finite_vals, vector_vals, 0.5);
  BOOST_CHECK_EQUAL(r.num_steps(), 2);
  BOOST_CHECK_EQUAL(r.size(), 6);
  BOOST_CHECK_EQUAL(r.weight(), 0.5);
  std::cout << r << std::endl;

  // test extract
  assignment a_all;
  r.extract(a_all);
  BOOST_CHECK_EQUAL(a_all.size(), 4);

  assignment a_cur;
  r.extract(0, a_cur);
  BOOST_CHECK_EQUAL(a_cur.size(), 2);

  hybrid_sequence_record<>::var_indices_type indices;
  indices.finite.push_back(std::make_pair(0, 0));
  indices.vector.push_back(std::make_pair(0, 1));
  hybrid_values<double> vals;
  r.extract(indices, vals);
  BOOST_CHECK_EQUAL(vals.finite.size(), 1);
  BOOST_CHECK_EQUAL(vals.vector.size(), 2);
}

BOOST_FIXTURE_TEST_CASE(test_load, fixture) {
  hybrid_sequence_record<> r(procs);
  std::vector<size_t> finite_vals(2, 1);
  std::vector<double> vector_vals(4, 2);
  r.assign(finite_vals, vector_vals, 0.5);
  
  proc_vector_type procs2(1, procs[0]);
  hybrid_sequence_record<> r2(procs2, 1.0);
  hybrid_sequence_record<>::proc_indices_type indices;
  indices.finite.push_back(0);
  r2.load(r, indices);

  BOOST_CHECK_EQUAL(r2.num_processes(), 1);
  BOOST_CHECK_EQUAL(r2.finite().num_processes(), 1); 
  BOOST_CHECK_EQUAL(r2.vector().num_processes(), 0);
  BOOST_CHECK_EQUAL(r2.num_steps(), 2);
  BOOST_CHECK_EQUAL(r2.weight(), 0.5);
}
