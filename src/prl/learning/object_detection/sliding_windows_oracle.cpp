#include <prl/learning/object_detection/sliding_windows_oracle.hpp>

#include <prl/macros_def.hpp>

namespace prl {

    void sliding_windows_oracle::init() {
      assert(window_h > 1 && window_w > 1);
      if (class_variable_ != NULL) {
        assert(label_ < class_variable_->size());
        add_finite_variable(class_variable_, true);
      }
      if (!(image_records_)) {
        non_image_rec =
          record(finite_numbering_ptr_, vector_numbering_ptr_, dvector);
//        if (class_variable_ != NULL)
//          non_image_rec.finite().back() = label_;
      }
      // Set scale_ so it skips scalings which won't affect image dimensions.
      double minscale(window_h < window_w ?
                      window_h / (window_h - 1) :
                      window_w / (window_w - 1));
      scale_ = std::max(scale_, minscale);
    }

    bool sliding_windows_oracle::next_image() {
      if (use_oracle) {
        do {
          if (!(img_o_ptr->next())) {
            if (auto_reset_) {
              img_o_ptr->reset();
              img_o_ptr->next();
            } else {
              return false;
            }
          }
        } while (image::true_height(img_o_ptr->current()) < window_h ||
                 image::true_width(img_o_ptr->current()) < window_w);
        current_rec = img_o_ptr->current();
      }
      current_scale = 1;
      max_scale = std::min(image::true_height(current_rec) / window_h,
                           image::true_width(current_rec) / window_w);
      current_row = 0; current_col = 0;
      if (class_variable_ != NULL) {
        current_rec.finite_numbering_ptr = finite_numbering_ptr_;
        current_rec.finite().push_back(label_);
      }
      return true;
    }

    // Public methods
    //==========================================================================

    const record& sliding_windows_oracle::current() const {
      if (image_records_) {
        return current_rec;
      } else {
        return non_image_rec;
      }
    }

    bool sliding_windows_oracle::next() {
      if (current_scale == 0) {
        if (next_image() == false)
          return false;
      } else {
        current_col += shift_;
        if (current_col + window_w > image::scaled_width(current_rec)) {
          current_col = 0;
          current_row += shift_;
        }
        if (current_row + window_h > image::scaled_height(current_rec)) {
          current_scale *= scale_;
          if (current_scale > max_scale) {
            if (use_oracle) {
              if (next_image() == false) {
                return false;
              }
            } else {
              if (!auto_reset_ || next_image() == false) {
                return false;
              }
            }
          }
          /*
            if ((!use_oracle && !auto_reset_) || next_image() == false)
            return false;
          */
          current_row = 0; current_col = 0;
        }
      }
      image::set_view(current_rec, current_row, current_col,
                      window_h, window_w,
                      1. / current_scale, 1. / current_scale);
      if (!(image_records_))
        image::get_simple_view(current_rec, non_image_rec, vector_seq);
      return true;
    }

} // namespace prl

#include <prl/macros_undef.hpp>
