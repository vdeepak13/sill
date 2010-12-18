#include <sill/factor/xml/any_factor.hpp>
#include <sill/factor/xml/constant_factor.hpp>
#include <sill/factor/xml/table_factor.hpp>

#include <sill/archive/xml_ofarchive.hpp>
#include <sill/archive/xml_iarchive.hpp>

int main()
{
  using namespace sill;

  typedef sill::any_factor<double> polymorphic;
  polymorphic::register_factor< tablef >();
  polymorphic::register_factor< constant_factor >();

  universe u;
  var_vector vf = u.new_finite_variables(2, 2);

  xml_ofarchive out("factors.xml");
  out << polymorphic(tablef(vf, 1));
  out << polymorphic(constant_factor(2));

  out.close();
}

