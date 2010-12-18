#ifndef SILL_SLIDING_WINDOWS_ORACLE_HPP
#define SILL_SLIDING_WINDOWS_ORACLE_HPP

#include <sill/learning/dataset/oracle.hpp>
#include <sill/learning/object_detection/image.hpp>
#include <sill/learning/object_detection/image_oracle.hpp>
#include <sill/math/free_functions.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Oracle which generates (optionally labeled) fixed-size images from
   * variable-sized images by choosing windows with different scalings and
   * locations.
   * This deterministically generates all windows at all scalings and positions.
   * It can either:
   *  - take a single image and generate its windows or
   *  - take an image oracle and generate all windows for all images.
   *
   * \author Joseph Bradley
   * \todo Support color images. (Does it not?)
   */
  class sliding_windows_oracle : public oracle {

    typedef oracle base;

    // Public types
    //==========================================================================
  public:

    struct parameters {

      //! If true, then this produces image records (See image.hpp.);
      //! if false, then it copies image views into normal records.
      //!  (default = true)
      bool image_records;

      //! Amount (> 1) by which to shrink (by dividing the width and height of)
      //! the image to examine the image at different scales.
      //!  (default = 1.25)
      double scale;

      //! Number of pixels (>= 1) by which to move the windows
      //! as they are slid across the image.
      //!  (default = 2)
      size_t shift;

      /**
       * If not NULL, then LABEL is attached to each image.
       * This class variable SHOULD NOT be in the image
       * records passed to this oracle or in the variable ordering.
       *  (default = NULL)
       */
      finite_variable* class_variable;

      //! Label used if CLASS_VARIABLE != NULL.
      //!  (default = 0)
      size_t label;

      //! If true, then automatically reset this oracle
      //! to loop through the image(s) again.
      //!  (default = false)
      bool auto_reset;

      parameters()
        : image_records(true), scale(1.25), shift(2), class_variable(NULL),
          label(0), auto_reset(false) { }

      bool valid() const {
        if (scale <= 1)
          return false;
        if (shift <= 1)
          return false;
        return true;
      }

    };  // struct parameters

    // Protected data members and methods
    //==========================================================================
  protected:

    // Copied from parameters:
    bool image_records_;
    double scale_;
    size_t shift_;
    finite_variable* class_variable_;
    size_t label_;
    bool auto_reset_;

    //! Window height (same as params)
    size_t window_h;

    //! Window width (same as params)
    size_t window_w;

    //! Indicates if images should be drawn from img_o_ptr.
    //! If false, then a single image (in current_rec) is used
    bool use_oracle;

    //! Image oracle
    copy_ptr<image_oracle> img_o_ptr;

    //! Current scale (true height / current_scale = scaled height)
    double current_scale;

    //! Max current_scale for current image
    double max_scale;

    //! Current row
    size_t current_row;

    //! Current column
    size_t current_col;

    //! Current image
    record current_rec;

    //! Record used if a non-image record is required
    record non_image_rec;

    //! Initialize oracle.
    void init();

    //! Goes to the next image, resetting current_* and scaled_*.
    //! Returns false if there are no more images (for image oracle),
    //! or resets the sliding_windows_oracle (for single image).
    bool next_image();

    // Constructors and destructors
    //==========================================================================
  public:
    /**
     * Constructor for non-randomized oracle which sequentially goes through
     * each image from the oracle and generates all windows for given
     * SCALE and SHIFT.
     *
     * @param img_o_ptr  image oracle
     * @param var_order  vector variables to be used for windows
     * @param window_h   height of windows
     * @param window_w   width of windows
     * @param params     parameters
     */
    sliding_windows_oracle
    (copy_ptr<image_oracle> img_o_ptr, const vector_var_vector& var_order,
     size_t window_h, size_t window_w, parameters params = parameters())
      : base(finite_var_vector(), var_order,
             std::vector<variable::variable_typenames>(var_order.size(),variable::VECTOR_VARIABLE)),
        image_records_(params.image_records), scale_(params.scale),
        shift_(params.shift), class_variable_(params.class_variable),
        label_(params.label), auto_reset_(params.auto_reset),
        window_h(window_h), window_w(window_w),
        use_oracle(true), img_o_ptr(img_o_ptr), current_scale(0) {
      assert(params.valid());
      init();
    }

    /**
     * Constructor for non-randomized oracle which sequentially goes through
     * the given image and generates all windows for given SCALE and SHIFT.
     *
     * @param img        image (This must be an image record.)
     * @param var_order  vector variables to be used for windows
     * @param window_h   height of windows
     * @param window_w   width of windows
     * @param params     parameters
     */
    sliding_windows_oracle
    (const record& img, const vector_var_vector& var_order,
     size_t window_h, size_t window_w, parameters params = parameters())
      : base(finite_var_vector(), var_order,
             std::vector<variable::variable_typenames>(var_order.size(),variable::VECTOR_VARIABLE)),
        image_records_(params.image_records), scale_(params.scale),
        shift_(params.shift), class_variable_(params.class_variable),
        label_(params.label), auto_reset_(params.auto_reset),
        window_h(window_h), window_w(window_w),
        use_oracle(false), current_scale(0), current_rec(img) {
      assert(image::true_height(img) >= window_h &&
             image::true_width(img) >= window_w);
      assert(params.valid());
      init();
    }

    // Public methods
    //==========================================================================

    //! Returns the current record.
    const record& current() const;

    //! Increments the oracle to the next record.
    //! This returns true iff the operation was successful; false indicates
    //! a depleted oracle.
    bool next();

  }; // class sliding_windows_oracle

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_SLIDING_WINDOWS_ORACLE_HPP
