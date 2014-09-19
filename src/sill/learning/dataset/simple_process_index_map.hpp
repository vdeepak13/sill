#ifndef SILL_SIMPLE_PROCESS_INDEX_MAP_HPP
#define SILL_SIMPLE_PROCESS_INDEX_MAP_HPP

#include <sill/base/discrete_process.hpp>
#include <sill/parsers/string_functions.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename Variable>
  class simple_process_index_map {
  public:
    typedef Variable                   variable_type;
    typedef discrete_process<Variable> process_type;

    simple_process_index_map() { }

    simple_process_index_map(const std::vector<process_type*>& procs) {
      initialize(procs);
    }

    void initialize(const std::vector<process_type*>& procs) {
      for (size_t i = 0; i < procs.size(); ++i) {
        map_.insert(std::make_pair(procs[i], i));
      }
    }

    void indices(const std::vector<process_type*>& procs,
                 std::vector<size_t>& result) const {
      result.reserve(procs.size());
      foreach(process_type* p, procs) {
        result.push_back(safe_get(map_, p));
      }
    }

    void indices(const std::vector<variable_type*>& vars,
                 std::vector<std::pair<size_t,size_t> >& result) const {
      return indices(vars, 0, result);
    }

    void indices(const std::vector<variable_type*>& vars,
                 size_t offset,
                 std::vector<std::pair<size_t,size_t> >& result) const {
      result.reserve(vars.size());
      foreach(variable_type* v, vars) {
        process_type* p = dynamic_cast<process_type*>(v->process());
        size_t i = safe_get(map_, p);
        int t = boost::any_cast<int>(v->index());
        if (t == current_step) {
          result.push_back(std::make_pair(i, offset + 0));
        } else if (t == next_step) {
          result.push_back(std::make_pair(i, offset + 1));
        } else if (t >= 0) {
          result.push_back(std::make_pair(i, offset + t));
        } else {
          throw std::invalid_argument("Invalid time step " + to_string(t));
        }
      }
    }

  private:
    std::map<process_type*, size_t> map_;

  }; // class simple_process_index_map

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
