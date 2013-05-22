#include <sill/base/universe.hpp>
#include <sill/base/variable_type_group.hpp>

template <typename DomainType>
void do_something();

template <>
void do_something<sill::finite_domain>() {
  using namespace sill;
  universe u;
  finite_domain d;
  d.insert(u.new_finite_variable(2));
  std::cerr << d << std::endl;
}

template <>
void do_something<sill::vector_domain>() {
  using namespace sill;
  universe u;
  vector_domain d;
  d.insert(u.new_vector_variable(2));
  std::cerr << d << std::endl;
}

int main(int argc, char** argv) {
  using namespace sill;

  do_something<variable_type_group<finite_variable>::domain_type>();
  do_something<variable_type_group<vector_variable>::domain_type>();

  return 0;
}
