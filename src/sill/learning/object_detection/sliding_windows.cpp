#include <sill/learning/object_detection/sliding_windows.hpp>
#include <sill/math/operations.hpp>

#include <sill/macros_def.hpp>

namespace sill {

    // Prediction methods
    //==========================================================================

    std::vector<boost::tuple<size_t, size_t, size_t, size_t> >
    sliding_windows::predict(record_type& example) const {
      std::vector<boost::tuple<size_t, size_t, size_t, size_t> > windows;
      for (double scale = 1; ; scale /= params.scale) {
        // newrows, newcols are height, width of scaled image
        size_t newrows = floor((double)(image::true_height(example)) * scale);
        size_t newcols = floor((double)(image::true_width(example)) * scale);
        if (newrows < window_h || newcols < window_w)
          break;
        // These settings for scaleh, scalew overcome numerical precision
        //  issues in image views; fix image views to be safer later
        //  (or switch completely to OpenCV images).
        double scaleh(((double)(newrows)/image::true_height(example)
                       + scale) / 2.);
        double scalew(((double)(newcols)/image::true_width(example)
                       + scale) / 2.);
        if (params.image_records) {
          for (size_t i = 0; i <= newrows - window_h; i += params.shift) {
            for (size_t j = 0; j < newcols - window_w; j += params.shift) {
              // look at window (i,j,window_h,window_w) (after scaling)
              image::set_view(example, i, j, window_h, window_w,
                              scaleh, scalew);
              if (base_classifier.predict_raw(example) > cutoff)
                windows.push_back
                  (boost::make_tuple
                   (round(i/scale), round(j/scale),
                    round(window_h/scale), round(window_w/scale)));
            }
          }
        } else {
          // Images passed to the base classifier must use the variables in
          // 'vector_seq' for pixel values.
          for (size_t i = 0; i <= newrows - window_h; i += params.shift) {
            for (size_t j = 0; j < newcols - window_w; j += params.shift) {
              // look at window (i,j,window_h,window_w) (after scaling)
              image::set_view(example, i, j, window_h, window_w,
                              scaleh, scalew);
              record_type ex2;
              image::get_simple_view(example, ex2, params.vector_seq);
              if (base_classifier.predict_raw(ex2) > cutoff)
                windows.push_back
                  (boost::make_tuple
                   (round(i/scale), round(j/scale),
                    round(window_h/scale), round(window_w/scale)));
            }
          }
        }
      }
      if (params.combine_overlapping) {
        assert(false);
      }
      return windows;
    }

    std::vector<sliding_windows::record_type>
    sliding_windows::intensity_maps(record_type& example) const {
      std::vector<record_type> pics;
      for (double scale = 1; ; scale /= params.scale) {
        // newrows, newcols are height, width of scaled image
        size_t newrows = floor((double)(image::true_height(example)) * scale);
        size_t newcols = floor((double)(image::true_width(example)) * scale);
        if (newrows < window_h || newcols < window_w)
          break;
        // These settings for scaleh, scalew overcome numerical precision
        //  issues in image views; fix image views to be safer later
        //  (or switch completely to OpenCV images).
        double scaleh(((double)(newrows)/image::true_height(example)
                       + scale) / 2.);
        double scalew(((double)(newcols)/image::true_width(example)
                       + scale) / 2.);
        image::set_view(example, 0, 0, newrows, newcols, scaleh, scalew);
        record_type pic(image::blank_image(example));
        size_t offset_h = (window_h - params.shift) / 2;
        size_t offset_w = (window_w - params.shift) / 2;
        if (params.image_records) {
          for (size_t i = 0; i <= newrows - window_h; i += params.shift) {
            for (size_t j = 0; j < newcols - window_w; j += params.shift) {
              // look at window (i,j,window_h,window_w) (after scaling)
              image::set_view(example, i, j, window_h, window_w,
                              scaleh, scalew);
              double conf = base_classifier.predict_raw(example);
              for (size_t i2 = i + offset_h;
                   i2 < i + offset_h + params.shift; ++i2)
                for (size_t j2 = j + offset_w;
                     j2 < j + offset_w + params.shift; ++j2)
                  image::set(pic, i2, j2, conf);
            }
          }
        } else {
          // TODO: THIS WOULD BE MORE EFFICIENT IF WE COMPUTED THE RESCALED
          //       IMAGE FIRST.
          // Images passed to the base classifier must use the variables in
          // 'vector_seq' for pixel values.
          for (size_t i = 0; i <= newrows - window_h; i += params.shift) {
            for (size_t j = 0; j < newcols - window_w; j += params.shift) {
              // look at window (i,j,window_h,window_w) (after scaling)
              image::set_view(example, i, j, window_h, window_w,
                              scaleh, scalew);
              record_type ex2;
              image::get_simple_view(example, ex2, params.vector_seq);
              double conf = base_classifier.predict_raw(ex2);
              for (size_t i2 = i + offset_h;
                   i2 < i + offset_h + params.shift; ++i2)
                for (size_t j2 = j + offset_w;
                     j2 < j + offset_w + params.shift; ++j2)
                  image::set(pic, i2, j2, conf);
            }
          }
        }
        pics.push_back(pic);
      }
      return pics;
    }

} // namespace sill

#include <sill/macros_undef.hpp>
