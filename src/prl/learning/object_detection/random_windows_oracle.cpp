#include <prl/learning/object_detection/random_windows_oracle.hpp>

#include <prl/macros_def.hpp>

namespace prl {

    // Protected data members and methods
    //==========================================================================

    void random_windows_oracle::init() {
      assert(images.size() > 0);
      rng.seed(static_cast<double>(params.random_seed));
      assert(window_h > 0 && window_w > 0);
      for (size_t i = 0; i < images.size(); ++i) {
        assert(image::height(images[i]) >= window_h);
        assert(image::width(images[i]) >= window_w);
      }
      if (params.class_variable != NULL) {
        assert(params.label < params.class_variable->size());
        add_finite_variable(params.class_variable, true);
      }
      if (!(params.image_records)) {
        current_rec =
          record(finite_numbering_ptr_, vector_numbering_ptr_, dvector);
      }
      if (params.class_variable != NULL) {
        size_t class_var_index = images.front().finite_numbering_ptr->size();
        for (size_t i = 0; i < images.size(); ++i) {
          if (images[i].finite_numbering_ptr->count(params.class_variable)) {
            std::cerr << "random_windows_oracle should not be told to set a "
                      << "class variable if the class variable is already in "
                      << "the image records." << std::endl;
            assert(false);
          }
          images[i].finite_numbering_ptr->operator[](params.class_variable) =
            class_var_index;
          images[i].finite().push_back(params.label);
        }
      }
    }

    // Public methods
    //==========================================================================

    const record& random_windows_oracle::current() const {
      if (params.image_records)
        return images[images_i];
      else
        return current_rec;
    }

    bool random_windows_oracle::next() {
      // Choose an image
      images_i = (size_t)(uniform_int(rng));
      record& img = images[images_i];
      // Choose a scaling
      double min_scale(std::max((double)(window_h) / image::true_height(img),
                                (double)(window_w) / image::true_width(img)));
      double scale_factor(boost::uniform_real<double>(min_scale,1)(rng));
      // Choose a position
      double row_factor(uniform_real(rng));
      double col_factor(uniform_real(rng));
      size_t scaled_height((size_t)
                           (floor(image::true_height(img) * scale_factor)));
      size_t scaled_width((size_t)
                          (floor(image::true_width(img) * scale_factor)));
      size_t r((size_t)(round((scaled_height - window_h) * row_factor)));
      size_t c((size_t)(round((scaled_width - window_w) * col_factor)));
      // Set the view
      image::set_view(images[images_i], r, c,
                      window_h, window_w,
                      scale_factor, scale_factor);
      if (!params.image_records)
        image::get_simple_view(images[images_i], current_rec,
                               vector_seq);
      return true;
    }

} // namespace prl

#include <prl/macros_undef.hpp>
