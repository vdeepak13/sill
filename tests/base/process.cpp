#define BOOST_TEST_MODULE process
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/base/discrete_process.hpp>
#include <sill/serialization/serialize.hpp>

using namespace sill;

struct fixture {
  fixture()
    : p(new finite_discrete_process("p", 4)),
      q(new finite_discrete_process("q", 2)) { }
  finite_discrete_process* p;
  finite_discrete_process* q;
};

BOOST_FIXTURE_TEST_CASE(test_construct, fixture) {
  BOOST_CHECK_EQUAL(p->name(), "p");
  BOOST_CHECK_EQUAL(p->size(), 4);
  BOOST_CHECK_EQUAL(q->name(), "q");
  BOOST_CHECK_EQUAL(q->size(), 2);
}

BOOST_FIXTURE_TEST_CASE(test_variables, fixture) {
  universe u;

  finite_variable* pt = p->current();
  finite_variable* pn = p->next();
  finite_variable* p1 = p->at(1);
  finite_variable* p2 = p->at(2);
  finite_variable* qt = q->current();
  finite_variable* q1 = q->at(1);
  
  BOOST_CHECK_EQUAL(pt->process(), p);
  BOOST_CHECK_EQUAL(pn->process(), p);
  BOOST_CHECK_EQUAL(p1->process(), p);
  BOOST_CHECK_EQUAL(p2->process(), p);
  BOOST_CHECK_EQUAL(qt->process(), q);
  BOOST_CHECK_EQUAL(q1->process(), q);

  BOOST_CHECK_EQUAL(boost::any_cast<int>(pt->index()), current_step);
  BOOST_CHECK_EQUAL(boost::any_cast<int>(pn->index()), next_step);
  BOOST_CHECK_EQUAL(boost::any_cast<int>(p1->index()), 1);
  BOOST_CHECK_EQUAL(boost::any_cast<int>(p2->index()), 2);
  BOOST_CHECK_EQUAL(boost::any_cast<int>(qt->index()), current_step);
  BOOST_CHECK_EQUAL(boost::any_cast<int>(q1->index()), 1);
  
  BOOST_CHECK_EQUAL(p1, p->at(1));

  std::set<finite_discrete_process*> procs = make_domain(p, q);
  finite_domain pqt = make_domain(pt, qt);
  finite_domain pq1 = make_domain(p1, q1);
  BOOST_CHECK_EQUAL(pqt, variables(procs, current_step));
  BOOST_CHECK_EQUAL(pq1, variables(procs, 1));
  BOOST_CHECK_EQUAL(procs, processes<finite_discrete_process>(pqt));
  BOOST_CHECK_EQUAL(procs, processes<finite_discrete_process>(pq1));
}

// TODO: serialization
