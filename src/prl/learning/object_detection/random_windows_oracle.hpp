#ifndef SILL_RANDOM_WINDOWS_ORACLE_HPP
#define SILL_RANDOM_WINDOWS_ORACLE_HPP
#include <map>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>

#include <sill/learning/dataset/oracle.hpp>
#include <sill/learning/object_detection/image.hpp>
#include <sill/math/free_functions.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Oracle which generates (optionally labeled) fixed-size images from
   * variable-sized images by choosing windows with different scalings and
   * locations.
   * This generates random windows from randomly chosen images.
   *
   * \author Joseph Bradley
   *
   * \todo Make this store the images it is given so it doesn't have to copy
   *       them on each next().
   */
  class random_windows_oracle : public oracle {

    typedef oracle base;

    // Public types
    //==========================================================================
  public:

    /**
     * PARAMETERS
     * \todo Make options for choosing how randomization is done.
     */
    struct parameters {

      //! If true, then this produces image records (See image.hpp.);
      //! if false, then it copies image views into normal records.
      //!  (default = true)
      bool image_records;

      //! Used to make the oracle deterministic
      //!  (default = time)
      double random_seed;

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

      parameters()
        : image_records(true), class_variable(NULL), label(0) {
        std::time_t time_tmp;
        time(&time_tmp);
        random_seed = time_tmp;
      }

    };  // struct parameters

    // Protected data members and methods
    //==========================================================================
  protected:

    parameters params;

    //! random number generator
    boost::mt11213b rng;

    //! Uniform [0,1]
    boost::uniform_real<double> uniform_real;

    //! Uniform [0,number of images - 1]
    boost::uniform_int<> uniform_int;

    //! Window height (same as params)
    size_t window_h;

    //! Window width (same as params)
    size_t window_w;

    //! Set of images
    std::vector<record> images;

    //! Index into images, pointing to current image
    size_t images_i;

    //! Current record (used when params.image_records == false)
    record current_rec;

    //! Initialize oracle.
    void init();

    // Constructors and destructors
    //==========================================================================
  public:

    /**
     * Constructor for randomized oracle which generates windows with random
     * scalings and positions from random images for the dataset.
     * NOTE: next() must be called to load the first record.
     * @param images     set of images
     * @param var_order  vector variables to be used for windows
     * @param window_h   height of windows
     * @param window_w   width of windows
     * @param params     parameters
     */
    random_windows_oracle
    (const std::vector<record>& images,
     const vector_var_vector& var_order,
     size_t window_h, size_t window_w, parameters params = parameters())
      : base(finite_var_vector(), var_order,
             std::vector<variable::variable_typenames>(var_order.size(),variable::VECTOR_VARIABLE)),
        params(params), uniform_real(0,1), uniform_int(0,images.size()-1),
        window_h(window_h), window_w(window_w), images(images), images_i(0),
        current_rec(finite_numbering_ptr_, vector_numbering_ptr_, dvector) {
      init();
    }

    // Public methods
    //==========================================================================

    //! Returns the current record.
    const record& current() const;

    //! Increments the oracle to the next record.
    //! This returns true iff the operation was successful; false indicates
    //! a depleted oracle.
    //! @todo This would probably be faster if it used uniform_int for
    //!       randomization for scaling and position.
    bool next();

  }; // class random_windows_oracle

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_RANDOM_WINDOWS_ORACLE_HPP
