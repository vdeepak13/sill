#include <sill/base/assignment.hpp>

namespace sill {
assignment map_union(const assignment& a, const assignment& b) {
  // Initialize the output map
  assignment output;
  output.finite() = map_union(a.finite(), b.finite());
  output.vector() = map_union(a.vector(), b.vector());
  return output;
}


assignment map_intersect(const assignment& a, const assignment& b) {
  // Initialize the output map
  assignment output;
  output.finite() = map_intersect(a.finite(), b.finite());
  output.vector() = map_intersect(a.vector(), b.vector());
  return output;
} 

assignment map_difference(const assignment& a, const assignment& b) {
  // Initialize the output map
  assignment output;
  output.finite() = map_difference(a.finite(), b.finite());
  output.vector() = map_difference(a.vector(), b.vector());
  return output;
}

std::ostream& operator<<(std::ostream& out, const assignment& a) {
  out << a.finite();
  out << a.vector();
  return out;
}

}
