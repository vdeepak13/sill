
#include <prl/base/universe.hpp>
#include <prl/base/variable_type_groups.hpp>

template <typename DomainType>
static void do_something();

template <>
static void do_something<prl::finite_domain>() {
  using namespace prl;
  universe u;
  finite_domain d;
  d.insert(u.new_finite_variable(2));
  std::cerr << d << std::endl;
}

template <>
static void do_something<prl::vector_domain>() {
  using namespace prl;
  universe u;
  vector_domain d;
  d.insert(u.new_vector_variable(2));
  std::cerr << d << std::endl;
}

int main(int argc, char** argv) {

  using namespace prl;

  do_something<variable_types<finite_variable>::domain_type>();

  do_something<variable_types<vector_variable>::domain_type>();

  return 0;
}
