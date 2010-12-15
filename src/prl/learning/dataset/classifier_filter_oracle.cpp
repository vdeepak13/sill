#include <prl/learning/dataset/classifier_filter_oracle.hpp>

#include <prl/macros_def.hpp>

namespace prl {

    // Mutating operations
    //==========================================================================

    bool classifier_filter_oracle::next() {
      count_ = 0;
      switch(mode) {
      case IS_VALUE:
        do {
          if (!(o.next()))
            return false;
          ++count_;
          if (count_ >= filter_limit)
            if (filter.predict(o.current()) == label_value)
              break;
            else
              return false;
        } while (filter.predict(o.current()) != label_value);
        break;
      case NOT_VALUE:
        do {
          if (!(o.next()))
            return false;
          ++count_;
          if (count_ >= filter_limit)
            if (filter.predict(o.current()) != label_value)
              break;
            else
              return false;
        } while (filter.predict(o.current()) == label_value);
        break;
      case ABOVE_THRESHOLD:
        do {
          if (!(o.next()))
            return false;
          ++count_;
          if (count_ >= filter_limit)
            if (filter.predict_raw(o.current()) > threshold)
              break;
            else
              return false;
        } while (filter.predict(o.current()) <= threshold);
        break;
      case BELOW_THRESHOLD:
        do {
          if (!(o.next()))
            return false;
          ++count_;
          if (count_ >= filter_limit)
            if (filter.predict_raw(o.current()) < threshold)
              break;
            else
              return false;
        } while (filter.predict(o.current()) >= threshold);
        break;
      case RIGHT_VALUE:
        do {
          if (!(o.next()))
            return false;
          ++count_;
          if (count_ >= filter_limit)
            if (filter.predict_raw(o.current()) ==
                o.current().finite(class_variable_index))
              break;
            else
              return false;
        } while (filter.predict_raw(o.current()) !=
                 o.current().finite(class_variable_index));
        break;
      case WRONG_VALUE:
        do {
          if (!(o.next()))
            return false;
          ++count_;
          if (count_ >= filter_limit)
            if (filter.predict_raw(o.current()) !=
                o.current().finite(class_variable_index))
              break;
            else
              return false;
        } while (filter.predict_raw(o.current()) ==
                 o.current().finite(class_variable_index));
        break;
      default:
        assert(false);
      }
//      current_record = o.current_record;
      /*
      if (o.current_record.own) {
        current_record.fin_ptr = &(o.current_record.findata);
        current_record.vec_ptr = &(o.current_record.vecdata);
      } else {
        current_record.fin_ptr = o.current_record.fin_ptr;
        current_record.vec_ptr = o.current_record.vec_ptr;
      }
      */
      return true;
    }

} // namespace prl

#include <prl/macros_undef.hpp>
