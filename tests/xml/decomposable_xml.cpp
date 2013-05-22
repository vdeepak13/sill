#include <sill/model/xml/decomposable.hpp>
#include <sill/factor/xml/table_factor.hpp>

#include <sill/archive/xml_ofarchive.hpp>
#include <sill/archive/xml_iarchive.hpp>

int main()
{
  using namespace sill;

  universe u;
  var_vector vf = u.new_finite_variables(2,2);

  std::vector< tablef > factors;
  factors.push_back(tablef(vf, 1));

  xml_ofarchive out("factors.xml");
  out << decomposable< tablef >(factors);

  out.close();
}

