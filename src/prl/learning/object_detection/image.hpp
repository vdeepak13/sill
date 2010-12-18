
#ifndef SILL_LEARNING_OBJECT_DETECTION_IMAGE_HPP
#define SILL_LEARNING_OBJECT_DETECTION_IMAGE_HPP

#include <map>

#include <sill/learning/dataset/data_loader.hpp>
#include <sill/math/free_functions.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // forward declaration
  class image_oracle;

  /**
   * Class for handling images.  This class defines a dataset representation
   * and provides an interface for working with grayscale and color images.
   *
   * This represents images as follows:
   *  - The initial vector variables give pixel values:
   *     - Each vector variable represents one pixel.
   *     - Vector variables are of dimension 1 for grayscale, 3 for RGB.
   *     - The natural ordering of the variables corresponds to the ordering
   *       of the pixels reading left-to-right, then top-to-bottom in the image
   *       (i.e., row-by-row).  This is the same ordering used by OpenCV.
   *  - Extra vector variables can go here and are not treated specially.
   *  - The last vector variable gives the image meta-data:
   *     - representation:
   *        - 0 = raw pixel values
   *        - 1 = integral representation
   *     - depth: number of values used to represent one pixel
   *       (1=grayscale, 3=color)
   *     - true_height: height of complete image
   *     - true_width: width of complete image
   *     - view_scaleh, view_scalew: in (0,1]
   *        (view_scaleh = height of each original pixel in scaled image view)
   *        - Note: The height of a view is floor(true_height * view_scaleh).
   *     - view_row, view_column: position of top-left corner of view
   *        w.r.t. scaling
   *     - view_height, view_width: dimensions of view w.r.t. scaling
   *  - Finite variables are not treated specially.
   *
   * \author Joseph Bradley
   * @todo Make functions for converting datasets which don't have all the
   *       metadata into datasets which do.
   */
  class image {

  private:
    static const bool debug = true;

    // Protected data members
    //==========================================================================
  protected:

    friend class image_oracle;
    friend class opencv_image_oracle;

    //! Constants for reading metadata
    enum metadata {HELLOHOWAREYOU, VIEW_WIDTH, VIEW_HEIGHT, VIEW_COL, VIEW_ROW,
                   VIEW_SCALEW, VIEW_SCALEH, TRUE_WIDTH, TRUE_HEIGHT,
                   DEPTH, REPRESENTATION, NMETAPLUS1};

    static const size_t NMETADATA = NMETAPLUS1 - 1;

    // Functions for reading metadata

    static size_t representation(const vec& vecdata) {
      return static_cast<size_t>(vecdata[vecdata.size() - REPRESENTATION]);
    }

    static size_t depth(const vec& vecdata) {
      return static_cast<size_t>(vecdata[vecdata.size() - DEPTH]);
    }

    static size_t true_height(const vec& vecdata) {
      return static_cast<size_t>(vecdata[vecdata.size() - TRUE_HEIGHT]);
    }

    static size_t true_width(const vec& vecdata) {
      return static_cast<size_t>(vecdata[vecdata.size() - TRUE_WIDTH]);
    }

    static size_t view_row(const vec& vecdata) {
      return static_cast<size_t>(vecdata[vecdata.size() - VIEW_ROW]);
    }

    static size_t view_col(const vec& vecdata) {
      return static_cast<size_t>(vecdata[vecdata.size() - VIEW_COL]);
    }

    static double view_scaleh(const vec& vecdata) {
      return vecdata[vecdata.size() - VIEW_SCALEH];
    }

    static double view_scalew(const vec& vecdata) {
      return vecdata[vecdata.size() - VIEW_SCALEW];
    }

    static size_t view_height(const vec& vecdata) {
      return static_cast<size_t>(vecdata[vecdata.size() - VIEW_HEIGHT]);
    }

    static size_t view_width(const vec& vecdata) {
      return static_cast<size_t>(vecdata[vecdata.size() - VIEW_WIDTH]);
    }

    //! Integral scaled height
    static size_t scaled_height(const vec& vecdata) {
      return static_cast<size_t>(floor(vecdata[vecdata.size() - TRUE_HEIGHT] *
                                       vecdata[vecdata.size() - VIEW_SCALEH]));
    }

    //! Integral scaled width
    static size_t scaled_width(const vec& vecdata) {
      return static_cast<size_t>(floor(vecdata[vecdata.size() - TRUE_WIDTH] *
                                       vecdata[vecdata.size() - VIEW_SCALEW]));
    }

    static double& representation(vec& vecdata) {
      return vecdata[vecdata.size() - REPRESENTATION];
    }

    static double& depth(vec& vecdata) {
      return vecdata[vecdata.size() - DEPTH];
    }

    static double& true_height(vec& vecdata) {
      return vecdata[vecdata.size() - TRUE_HEIGHT];
    }

    static double& true_width(vec& vecdata) {
      return vecdata[vecdata.size() - TRUE_WIDTH];
    }

    static double& view_row(vec& vecdata) {
      return vecdata[vecdata.size() - VIEW_ROW];
    }

    static double& view_col(vec& vecdata) {
      return vecdata[vecdata.size() - VIEW_COL];
    }

    static double& view_scaleh(vec& vecdata) {
      return vecdata[vecdata.size() - VIEW_SCALEH];
    }

    static double& view_scalew(vec& vecdata) {
      return vecdata[vecdata.size() - VIEW_SCALEW];
    }

    static double& view_height(vec& vecdata) {
      return vecdata[vecdata.size() - VIEW_HEIGHT];
    }

    static double& view_width(vec& vecdata) {
      return vecdata[vecdata.size() - VIEW_WIDTH];
    }

    // Publics methods: getting image info
    //==========================================================================
  public:

    //! Return height of image (or height of view) with respect to scaled
    //! width/height
    static size_t height(const record& r) {
      return view_height(r.vector());
    }

    //! Return width of image (or width of view) with respect to scaled
    //! width/height
    static size_t width(const record& r) {
      return view_width(r.vector());
    }

    //! Return actual height of image (regardless of view)
    static size_t true_height(const record& r) {
      return true_height(r.vector());
    }

    //! Return actual width of image (regardless of view)
    static size_t true_width(const record& r) {
      return true_width(r.vector());
    }

    //! Return top-left corner (row,col) of view (or (0,0) if no view)
    //! with respect to scaled width/height
    static std::pair<size_t,size_t> view_pos(const record& r) {
      return std::make_pair(view_row(r.vector()),view_col(r.vector()));
    }

    //! Return the scaled height of the view.
    static size_t scaled_height(const record& r) {
      return scaled_height(r.vector());
    }

    //! Return the scaled width of the view.
    static size_t scaled_width(const record& r) {
      return scaled_width(r.vector());
    }

    //! Return element (row,col) in image (or in view) with respect to scaled
    //! width/height
    //! NOTE: If the image is scaled, this returns a pixel in approximately
    //!       the correct location, but it does not necessarily return the
    //!       same pixel value as get_simple_view().
    //! @todo This is currently a bottleneck for haar.  Optimize it.
    //! Note: This does not do bound checking!
    static double get_pixel(const record& r, size_t row, size_t col);

    //! Return variable corresponding to element (row,col) in image
    //! which is not a view and is not scaled.
    static vector_variable* get_var(const vector_var_vector& vector_seq,
                                    size_t width, size_t row, size_t col) {
      return vector_seq[width * row + col];
    }

    /**
     * Returns a record of the view only, without the extra image metadata.
     * The pixel value variables will be the first set of variables from the
     * original pixel value variables, and the non-pixel variables will
     * be the same and in the natural order.
     * Non-image data in vector and finite variables is retained.
     * @param r     Image record.
     * @param newr  Non-image record in which data will be stored.
     */
    static void get_simple_view(const record& r, record& newr);

    /**
     * Returns a record of the view only, without the extra image metadata,
     * using the given variables.
     * Non-image data in vector and finite variables is retained.
     * @param r          Image record.
     * @param newr       Non-image record in which data will be stored.
     * @param var_order  Vector variables to use in the returned record for
     *                   the pixel values (with one variable per pixel).
     *                   This should not include the other vector variables or
     *                   finite variables.
     */
    static void get_simple_view(const record& r, record& newr,
                                vector_var_vector var_order);


    //! Create a blank image with the same dimensions as the given image
    //! (using the view height and width).  This uses variables from img
    //! for the new image.
    static record blank_image(const record& img);

    //! Sets element (row,col) in image (or in view).
    //! \todo This does not support scaling yet.
    static void set(record& r, size_t row, size_t col, double val);

    //! Modify image in raw format to be in integral representation
    static void raw2integral(record& r);

    //! Modify image in raw format to be in integral representation
    static record raw2integral(const record& r);

    //! Reset view of image to the entire image
    static void reset_view(record& r);

    //! Set view of image to upper-left corner (row,col), height h, width w,
    //! with scaling scaleh, scalew
    static void set_view(record& r, size_t row, size_t col, size_t h, size_t w,
                         double scaleh, double scalew);

    //! Writes an image (or view) to the given output stream as a matrix.
    template <typename CharT, typename Traits>
    static void write(std::basic_ostream<CharT, Traits>& out,
                      const record& r) {
      const vec& vecdata = r.vector();
      if (depth(vecdata) != 1) {
        if (vecdata.size() < 20)
          std::cerr << vecdata << std::endl;
        std::cerr << "image::write not implemented for depth = "
                  << depth(vecdata) << " yet."
                  << std::endl;
        assert(false);
      }
      size_t cur_width = out.width();
      size_t cur_prec = out.precision();
      write_metadata(out, r);
      out << std::endl;
      if (view_scaleh(vecdata) == 1 && view_scalew(vecdata) == 1) {
        // no scaling required
        for (size_t i = 0; i < view_height(vecdata); ++i) {
          for (size_t j = 0; j < view_width(vecdata); ++j) {
            out.width(8);
            out.precision(6);
            out << vecdata[true_width(vecdata) * (view_row(vecdata) + i)
                           + view_col(vecdata) + j]
                << " ";
          }
          out << "\n";
        }
      } else {
        size_t sheight = view_height(vecdata);
        size_t swidth = view_width(vecdata);
        record r2;
        get_simple_view(r, r2);
        const vec& vecdata2 = r2.vector();
        for (size_t i = 0; i < sheight; ++i) {
          for (size_t j = 0; j < swidth; ++j) {
            out.width(8);
            out.precision(6);
            out << vecdata2[swidth * i + j] << " ";
          }
          out << "\n";
        }
      }
      out.width(cur_width);
      out.precision(cur_prec);
    }

    //! Writes an image's metadata to the given output stream.  For debugging.
    template <typename CharT, typename Traits>
    static void write_metadata(std::basic_ostream<CharT, Traits>& out,
                               const record& r) {
      const vec& vecdata = r.vector();
      out << "representation=" << representation(vecdata)
          << ", true_height=" << true_height(vecdata)
          << ", true_width=" << true_width(vecdata)
          << ", view_row=" << view_row(vecdata)
          << ", view_col=" << view_col(vecdata)
          << ", view_height=" << view_height(vecdata)
          << ", view_width=" << view_width(vecdata)
          << ", view_scaleh=" << view_scaleh(vecdata)
          << ", view_scalew=" << view_scalew(vecdata);
    }

    //! Creates a var_order which works with images of the given dimensions.
    //! @param depth   1 = grayscale (default), 3 = color
    //! @param extra   extra vector variables to include
    static vector_var_vector
    create_var_order(universe& u, size_t h, size_t w, size_t depth,
                     const vector_var_vector& extra);

    // Publics methods: image datasets
    //==========================================================================

    /**
     * Read in dataset with grayscale or color images of equal size.
     * Each record must have vector variables in this order:
     *  - height
     *  - width
     *  - pixel values
     * All non-class finite variables are ignored.
     */
    template <typename MutableDataset>
    boost::shared_ptr<MutableDataset>
    read_dataset(const std::string& summary_filename, universe& u) {
      concept_assert((sill::MutableDataset<MutableDataset>));
      symbolic_oracle o(data_loader::load_symbolic_oracle(summary_filename, u));
      boost::shared_ptr<MutableDataset>
        ds_ptr(new MutableDataset(o.datasource_info()));
      assert(false); // NOT FINISHED
    }

  }; // class image

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_OBJECT_DETECTION_IMAGE_HPP
