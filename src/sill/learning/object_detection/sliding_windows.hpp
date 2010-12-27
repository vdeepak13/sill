
#ifndef SILL_LEARNING_DISCRIMINATIVE_SLIDING_WINDOWS_HPP
#define SILL_LEARNING_DISCRIMINATIVE_SLIDING_WINDOWS_HPP

#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>

#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/learning/dataset/record.hpp>
#include <sill/learning/discriminative/binary_classifier.hpp>
#include <sill/learning/object_detection/image.hpp>
#include <sill/math/free_functions.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

/**
 * \file sliding_windows.hpp Code for object detection in images using
 *                           sliding windows.
 */

namespace sill {

  /**
   * Class which takes a binary confidence-rated classifier for fixed-size
   * images and turns it into a classifier which takes larger images and finds
   * objects in those images.
   * The base classifier must be trained on images in which
   * the objects of interest are essentially filling the positive example
   * images (with some background being OK).  This classifier takes a new image
   * and scans over it, looking at windows (rectangular sections) at multiple
   * scales.  It returns a set of windows in which objects are detected with
   * high confidence, optionally combining overlapping windows.
   *
   * This assumes images are represented as in the image class.
   * It assumes the only finite variable is the binary class variable.
   *
   * \author Joseph Bradley
   * @see image.hpp
   */
  class sliding_windows {

    static const bool debug = true;

    // Public types
    //==========================================================================
  public:

    struct parameters {

      /**
       * IMAGE_RECORDS (bool): if true, then it assumes base classifier
       *  uses image records (See image.hpp.); if false, then it copies
       *  image views into normal records.
       *  (default = true)
       */
      bool image_records;

      /**
       * VECTOR_SEQ (vector_var_vector): If IMAGE_RECORDS == false, then this
       * is used for the vector variables in the records passed to the base
       * classifiers (1 for each pixel).
       * (required iff IMAGE_RECORDS == false)
       */
      vector_var_vector vector_seq;

      /**
       * SCALE (double): amount (> 1) by which to shrink (by dividing
       *  the width and height of) the image to examine the image at different
       *  scales
       *  (default = 1.25)
       */
      double scale;

      /**
       * SHIFT (size_t): number of pixels (>= 1) by which to move the windows
       *  as they are slid across the image
       *  (default = 2)
       */
      size_t shift;

      /**
       * COMBINE_OVERLAPPING (bool): if true, then this combines overlapping
       *  windows by averaging their corners
       *  (default = true)
       */
      bool combine_overlapping;

      /**
       * REQUIRED_OVERLAPPING (size_t): number of overlapping windows
       *  required to decide that there is an object
       *  (ignored if COMBINE_OVERLAPPING = false)
       *  (default = 2)
       *  TODO: IMPLEMENT THIS.
       */
      size_t required_overlapping;

      parameters()
        : image_records(true), scale(1.25), shift(2),
          combine_overlapping(true), required_overlapping(2) {
      }

      //! @param window_size  number of pixels per window
      bool valid(size_t window_size) const {
        if (!image_records)
          if (vector_seq.size() != window_size)
            return false;
        if (scale <= 1)
          return false;
        if (shift <= 1)
          return false;
        return true;
      }

    }; // class parameters

    // Protected data members and methods
    //==========================================================================
  protected:

    parameters params;

    //! Class variable
    finite_variable* class_variable;

    //! Index of class variable in records
    size_t class_variable_index;

    //! Base classifier
    const binary_classifier& base_classifier;

    //! If the base classifier's confidence is above this value for a window,
    //! then the object has been detected in the window.
    double cutoff;

    //! Height of window
    size_t window_h;

    //! Width of window
    size_t window_w;

//    //! Record used to store the sliding window
//    record window_record;

    // Constructors and destructors
    //==========================================================================
  public:

    /**
     * Constructor for a sliding windows classifier.
     *
     * @param base_classifier  base classifier
     * @param cutoff  If the base classifier's confidence is above this value
     *                for a window, then the object has been detected in the
     *                window.
     * @todo pass window_h, window_w in a better way
     */
    sliding_windows(const binary_classifier& base_classifier,
                    double cutoff, size_t window_h, size_t window_w,
                    parameters params = parameters())
      : params(params), class_variable(base_classifier.label()),
        class_variable_index(base_classifier.label_index()),
        base_classifier(base_classifier), cutoff(cutoff),
        window_h(window_h), window_w(window_w) {
      if (!params.valid(window_h * window_w))
        assert(false);
    }

    // Getters
    //==========================================================================
    /*
    //! Returns the finite variables in their natural order
    //! (including class variable).
    const finite_var_vector& finite_list() const {
      return base_classifier.finite_list();
    }

    //! Returns the vector variables in their natural order
    const vector_var_vector& vector_list() const {
      return base_classifier.vector_list();
    }
    */
    //! Returns the class variable
    finite_variable* label() const {
      return class_variable;
    }

    // Prediction methods
    //==========================================================================

    /**
     * Returns a set of windows indicating objects found in the image.
     * Each component of the returned vector is a tuple <x,y,h,w> where
     * x = left side x-coordinate, y = top y-coordinate, h = height in pixels,
     * w = width in pixels (indexing coordinates from 0, from upper-left).
     *
     * @param example  This must be an image record.
     * @see image.hpp
     */
    std::vector<boost::tuple<size_t, size_t, size_t, size_t> >
    predict(const record& example) const {
      record r(example);
      return predict(r);
    }

    /**
     * Returns a set of windows indicating objects found in the image.
     * Each component of the returned vector is a tuple <x,y,h,w> where
     * x = left side x-coordinate, y = top y-coordinate, h = height in pixels,
     * w = width in pixels (indexing coordinates from 0, from upper-left).
     *
     * @param example  This must be an image record.
     * @see image.hpp
     */
    std::vector<boost::tuple<size_t, size_t, size_t, size_t> >
    predict(record& example) const;

    /**
     * Returns a vector of intensity maps (as images) indicating where objects
     * are in the image.
     *
     * @param example  This must be an image record.
     * @see image.hpp
     */
    std::vector<record>
    intensity_maps(const record& example) const {
      record r(example);
      return intensity_maps(r);
    }

    /**
     * Returns a vector of intensity maps (as images) indicating where objects
     * are in the image.
     *
     * @param example  This must be an image record.
     * @see image.hpp
     */
    std::vector<record>
    intensity_maps(record& example) const;

  };  // class sliding_windows

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_SLIDING_WINDOWS_HPP
