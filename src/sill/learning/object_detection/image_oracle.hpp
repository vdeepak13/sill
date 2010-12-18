#ifndef SILL_IMAGE_ORACLE_HPP
#define SILL_IMAGE_ORACLE_HPP

#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <sill/global.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/base/universe.hpp>
#include <sill/learning/object_detection/image.hpp>
#include <sill/copy_ptr.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for loading variable-size images using the text format below.
   * This loads the images into image records for which the variables have
   * little meaning.  (See the image class.)
   *
   * Text image format for variable-size grayscale images:
   *  - Lines beginning with pipes ('|') will be ignored as comments.
   *  - Lines beginning with stars ('*') will be parsed as options;
   *    options should have format: *OPTION_NAME=OPTION_VALUE
   *     - fixed_size: tells the oracle if all images are of fixed size
   *                    (in which case it can be more efficient)
   *                   (default = false)
   *  - Comments and options MUST come before any images.
   *  - Each non-comment/option pair of lines contains:
   *     image_name
   *     height width [whitespace-separated list of pixel values]
   *  - Empty lines may be inserted anywhere.
   *  - The pixel values should be given row-by-row.
   *
   * \author Joseph Bradley
   * @see image.hpp
   */
  class image_oracle {

    // Protected data members
    //==========================================================================
  protected:

    //! Filename of dataset
    std::string data_filename;

    //! Fixed size images
    bool fixed_size;

    //! Input file stream of file currently being read.
    //! This is declared mutable b/c it's necessary for the copy constructor
    //! (which must call f_in.tellg()).
    mutable std::ifstream f_in;

    //! Temporaries used to avoid repeated allocation.
    std::string line;
    std::istringstream is;

    //! Current image's name
    std::string current_name_;

    //! Current image's record
    record current_rec;

    //! Universe
    universe& u;

    //! Variable for image metadata.
    vector_variable* metadata_var;

    //! Variables available for use in creating records.
    vector_var_vector vars;

    //! Temporary for vector numberings.
    //! This always holds the first (#) elements of vars.
    copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr;

    // Protected methods
    //==========================================================================

    void init();

    // Constructors and destructors
    //==========================================================================
  public:

    /**
     * Constructor for an image oracle.
     * NOTE: next() must be called to load the first record.
     * @param filename    data file name
     * @param u           universe in which to create new variables
     */
    image_oracle(const std::string& filename, universe& u)
      : data_filename(filename), fixed_size(false), u(u),
        metadata_var(u.new_vector_variable(image::NMETADATA)),
        vector_numbering_ptr(new std::map<vector_variable*, size_t>()) {
      init();
    }

    //! Copy constructor
    image_oracle(const image_oracle& o)
      : data_filename(o.data_filename), fixed_size(o.fixed_size),
        line(o.line), is(),
        current_rec(o.current_rec), u(o.u), metadata_var(o.metadata_var),
        vars(o.vars), vector_numbering_ptr(o.vector_numbering_ptr) {
      f_in.open(data_filename.c_str());
      assert(f_in.good());
      f_in.seekg(o.f_in.tellg());
    }

    //! Assignment operator
    image_oracle& operator=(const image_oracle& o);

    //! Destructor.
    ~image_oracle() {
      f_in.close();
    }

    // TODO: may need to implement operator= too b/c of variable 'f_in'

    // Public methods
    //==========================================================================

    //! Returns the current image's name.
    const std::string& current_name() const {
      return current_name_;
    }

    //! Returns the current image's record.
    const record& current() const {
      return current_rec;
    }

    //! Increments the oracle to the next record.
    //! This returns true iff the operation was successful; false indicates
    //! a depleted oracle.
    bool next();

    //! Resets the oracle, returning true if it has been reset.
    bool reset();

  }; // class image_oracle

  // Free functions
  //==========================================================================

  /**
   * Load a set of variable-size grayscale images from a data file.
   * @see image.hpp
   */
  boost::shared_ptr<std::vector<record> >
  load_images(const std::string& filename, universe& u);

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_IMAGE_ORACLE_HPP
