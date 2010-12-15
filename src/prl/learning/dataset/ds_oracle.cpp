#include <prl/learning/dataset/ds_oracle.hpp>

#include <prl/macros_def.hpp>

namespace prl {

    // Mutating operations
    //==========================================================================

    bool ds_oracle::next() {
      if (params.record_limit != 0 && records_used >= params.record_limit)
        return false;
      if (!initialized) {
        ds_it = ds.begin();
        if (ds_it == ds_end)
          return false;
        initialized = true;
      } else {
        ++ds_it;
        if (ds_it == ds_end) {
          if (params.auto_reset) {
            ds_it = ds.begin();
          }
          else {
            return false;
          }
        }
      }
      ds_it.load_cur_record();
//      current_record = ds_it.r;
      //      current_record = ds_it.get_record_ref();
      ++records_used;
      return true;
    }

} // namespace prl

#include <prl/macros_undef.hpp>
