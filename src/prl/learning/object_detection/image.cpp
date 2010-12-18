#include <map>

#include <sill/base/universe.hpp>
#include <sill/learning/object_detection/image.hpp>
#include <sill/base/stl_util.hpp>

#include <sill/macros_def.hpp>

namespace sill {

    // Publics methods: getting image info
    //==========================================================================
  
    double image::get_pixel(const record& r, size_t row, size_t col) {
      const vec& vecdata = r.vector();
//      assert(row < view_height(vecdata) && col < view_width(vecdata));
      if (view_scalew(vecdata) != 1 || view_scaleh(vecdata) != 1)
        return vecdata[(size_t)
                       (true_width(vecdata) *
                        round((view_row(vecdata)+row) / view_scaleh(vecdata)) +
                        round((view_col(vecdata)+col) / view_scalew(vecdata)))];
      else
        return vecdata[true_width(vecdata) * (view_row(vecdata) + row)
                       + view_col(vecdata) + col];
    }

    void image::get_simple_view(const record& r, record& newr) {
      const vec& vecdata = r.vector();
      assert(vecdata.size() >=
             true_height(vecdata) * true_width(vecdata) + NMETADATA);
      vector_var_vector vector_list(r.vector_list());
      vector_var_vector var_order;
      assert(vector_list.size() >= view_height(vecdata) * view_width(vecdata));
      for (size_t j = 0; j < view_height(vecdata) * view_width(vecdata); ++j)
        var_order.push_back(vector_list[j]);
      get_simple_view(r, newr, var_order);
    }

    void image::get_simple_view(const record& r, record& newr,
                                vector_var_vector var_order) {
      vector_var_vector vector_list(r.vector_list());
      const vec& vecdata = r.vector();
      size_t orig_image_dim = true_height(vecdata) * true_width(vecdata);
      size_t orig_image_size = orig_image_dim * depth(vecdata);
      size_t new_image_dim = view_height(vecdata) * view_width(vecdata);
      size_t new_image_size = new_image_dim * depth(vecdata);
      assert(vecdata.size() >= orig_image_size + NMETADATA);
      size_t n_extra_vecs = vector_list.size() - (orig_image_dim + 1);
      size_t extra_vecs_size = 0;

      // prepare vector_numbering (finite_numbering is the same)
      copy_ptr<std::map<vector_variable*, size_t> >
        vector_numbering_ptr(new std::map<vector_variable*, size_t>());
      assert(var_order.size() == new_image_dim);
      for (size_t j = 0; j < new_image_dim; ++j) {
        assert(var_order[j]->size() == depth(vecdata));
        vector_numbering_ptr->operator[](var_order[j]) = j * depth(vecdata);
      }
      for (size_t j = 0; j < n_extra_vecs; ++j) {
        assert(vector_list.size() > orig_image_dim + j);
        extra_vecs_size += vector_list[orig_image_dim + j]->size();
        vector_numbering_ptr->operator[](vector_list[orig_image_dim + j])
          = new_image_size + j;
      }
      assert(vecdata.size() == orig_image_size + extra_vecs_size + NMETADATA);

      newr.reset(r.finite_numbering_ptr, vector_numbering_ptr,
                 new_image_size + extra_vecs_size);

      // prepare finite and vector values
      vec vec_val(new_image_size + extra_vecs_size, 0);
      if (view_scaleh(vecdata) == 1 && view_scalew(vecdata) == 1) {
        // no scaling required
        for (size_t h = 0; h < view_height(vecdata); ++h) {
          size_t offset1 = depth(vecdata) *
            (true_width(vecdata) * (view_row(vecdata)+h) + view_col(vecdata));
          size_t offset2 = h * view_width(vecdata) * depth(vecdata);
          for (size_t w = 0; w < view_width(vecdata); ++w) {
            for (size_t c = 0; c < depth(vecdata); ++c) {
              vec_val[offset2] = vecdata[offset1];
              ++offset1;
              ++offset2;
            }
          }
        }
      } else {
        // TODO: I DON'T HANDLE END CASES CAREFULLY FOR SCALING, SO THERE IS
        //       SOME DISTORTION AT THE EDGES OF IMAGES.

        // scale image down by multiplicative view_scaleh, view_scalew < 1
        // foreach row i of the original image
        //  which contributes to row indi1 (and maybe indi2) in the new image
        for (size_t i = 0; i < view_height(vecdata)/view_scaleh(vecdata); ++i) {
          size_t orig_offset = depth(vecdata)
            * (true_width(vecdata) * (view_row(vecdata) + i)
               + view_col(vecdata));
          double indi = view_scaleh(vecdata) * i;
          double end_adjust_i =
            std::min(1.,
                     (double)(view_height(vecdata)) / view_scaleh(vecdata) - i);
          size_t indi1 = (size_t)floor(indi);
          size_t indi2 = indi1 + 1;
          double indi1_weight = std::min(indi2 - indi, view_scaleh(vecdata));
          double indi2_weight =
            end_adjust_i * view_scaleh(vecdata) - indi1_weight;
          size_t new_offset = indi1 * view_width(vecdata) * depth(vecdata);
          for (size_t j = 0; j < view_width(vecdata) / view_scalew(vecdata);
               ++j) {
            double indj = view_scalew(vecdata) * j;
            double end_adjust_j =
              std::min(1.,
                       (double)(view_width(vecdata))/view_scalew(vecdata) - j);
            size_t indj1 = (size_t)floor(indj);
            size_t indj2 = indj1 + 1;
            double indj1_weight = std::min(indj2 - indj, view_scalew(vecdata));
            double indj2_weight =
              end_adjust_j * view_scalew(vecdata) - indj1_weight;
            for (size_t c = 0; c < depth(vecdata); ++c) {
              vec_val[new_offset + indj1 * depth(vecdata) + c]
                += indi1_weight * indj1_weight * vecdata[orig_offset];
              if (indj2_weight > 0)
                vec_val[new_offset + indj2 * depth(vecdata) + c]
                  += indi1_weight * indj2_weight * vecdata[orig_offset];
              ++orig_offset;
            }
          }
          if (indi2_weight > 0) {
            size_t orig_offset = depth(vecdata)
              * (true_width(vecdata) * (view_row(vecdata) + i)
                 + view_col(vecdata));
            size_t new_offset =
              indi2 * view_width(vecdata) * depth(vecdata);
            for (size_t j = 0; j < view_width(vecdata) / view_scalew(vecdata);
                 ++j) {
              double indj = view_scalew(vecdata) * j;
              double end_adjust_j =
                std::min(1.,
                         (double)(view_width(vecdata))/view_scalew(vecdata)-j);
              size_t indj1 = (size_t)floor(indj);
              size_t indj2 = indj1 + 1;
              double indj1_weight = std::min(indj2-indj, view_scalew(vecdata));
              double indj2_weight =
                end_adjust_j * view_scalew(vecdata) - indj1_weight;
              for (size_t c = 0; c < depth(vecdata); ++c) {
                vec_val[new_offset + indj1 * depth(vecdata) + c]
                  += indi2_weight * indj1_weight * vecdata[orig_offset];
                if (indj2_weight > 0)
                  vec_val[new_offset + indj2 * depth(vecdata) + c]
                    += indi2_weight * indj2_weight * vecdata[orig_offset];
                ++orig_offset;
              }
            }
          }
        }
      }
      newr.set_finite_val(r.finite());
      newr.set_vector_val(vec_val);
    }

    // Publics methods: modifying images
    //==========================================================================

    record image::blank_image(const record& img) {
      const vec& vecdata = img.vector();
      copy_ptr<std::map<vector_variable*, size_t> >
        vec_numbering_ptr(new std::map<vector_variable*, size_t>());
      assert(true_height(vecdata) * true_width(vecdata) + 1
             == img.vector_numbering_ptr->size());
      assert(img.finite_numbering_ptr->size() == 0);
      size_t pixelsize
        = (vecdata.size() - NMETADATA) / (img.vector_numbering_ptr->size() - 1);
      size_t imgsize = true_height(vecdata) * true_width(vecdata);
      size_t newimgsize = view_height(vecdata) * view_width(vecdata);
      for (std::map<vector_variable*, size_t>::const_iterator it
             = img.vector_numbering_ptr->begin();
           it != img.vector_numbering_ptr->end(); ++it)
        if (it->second < newimgsize)
          vec_numbering_ptr->operator[](it->first) = it->second;
        else if (it->second == imgsize)
          vec_numbering_ptr->operator[](it->first) = newimgsize;
      record newimg(img.finite_numbering_ptr, vec_numbering_ptr,
                    newimgsize * pixelsize + NMETADATA);
      vec& newvec = newimg.vector();
      representation(newvec) = 0;
      depth(newvec) = depth(vecdata);
      true_height(newvec) = view_height(vecdata);
      true_width(newvec) = view_width(vecdata);
      view_row(newvec) = 0;
      view_col(newvec) = 0;
      view_height(newvec) = view_height(vecdata);
      view_width(newvec) = view_width(vecdata);
      view_scaleh(newvec) = 1;
      view_scalew(newvec) = 1;
      return newimg;
    }

    void image::set(record& r, size_t row, size_t col, double val) {
      vec& vecdata = r.vector();
      assert(row < view_height(vecdata) && col < view_width(vecdata));
      assert(view_scaleh(vecdata) == 1 && view_scalew(vecdata) == 1);
      vecdata[(size_t)(true_width(vecdata) * (view_row(vecdata) + row)
                       + view_col(vecdata) + col)] = val;
    }

    void image::raw2integral(record& r) {
      vec& vecdata = r.vector();
      assert(representation(vecdata) == 0);
      representation(vecdata) = 1;
      assert(vecdata.size() >=
             true_height(vecdata) * true_width(vecdata) + NMETADATA);
      std::vector<size_t> offsets((size_t)true_width(vecdata));
      for (size_t i = 0; i < true_height(vecdata); ++i)
        offsets[i] = (size_t)(i * true_width(vecdata) * depth(vecdata));
      for (size_t i = 1; i < true_height(vecdata); ++i)
        for (size_t c = 0; c < depth(vecdata); ++c)
          vecdata[offsets[i]+c] += vecdata[offsets[i-1]+c];
      for (size_t j = 1 * (size_t)depth(vecdata);
           j < true_width(vecdata) * depth(vecdata); ++j)
        vecdata[j] += vecdata[j - (size_t)depth(vecdata)];
      for (size_t i = 1; i < true_height(vecdata); ++i)
        for (size_t j = 1 * (size_t)depth(vecdata);
             j < true_width(vecdata) * depth(vecdata); ++j)
          vecdata[offsets[i]+j] +=
            vecdata[offsets[i-1]+j]
            + vecdata[offsets[i]+j-(size_t)depth(vecdata)]
            - vecdata[offsets[i-1]+j-(size_t)depth(vecdata)];
    }

    record image::raw2integral(const record& r) {
      record r2(r.finite_numbering_ptr, r.vector_numbering_ptr,
                r.vector().size());
      raw2integral(r2);
      return r2;
    }

    void image::reset_view(record& r) {
      vec& vecdata = r.vector();
      view_row(vecdata) = 0;
      view_col(vecdata) = 0;
      view_height(vecdata) = true_height(vecdata);
      view_width(vecdata) = true_width(vecdata);
      view_scaleh(vecdata) = 1;
      view_scalew(vecdata) = 1;
    }

    void image::set_view(record& r, size_t row, size_t col, size_t h, size_t w,
                         double scaleh, double scalew) {
      vec& vecdata = r.vector();
      assert(h > 0 && w > 0);
      assert(scaleh > 0 && scaleh <= 1 && scalew > 0 && scalew <= 1);
      view_row(vecdata) = row; view_col(vecdata) = col;
      view_height(vecdata) = h; view_width(vecdata) = w;
      view_scaleh(vecdata) = scaleh; view_scalew(vecdata) = scalew;
      assert(row + h - 1 < scaled_height(vecdata) &&
             col + w - 1 < scaled_width(vecdata));
    }

    vector_var_vector
    image::create_var_order(universe& u, size_t h, size_t w, size_t depth,
                     const vector_var_vector& extra) {
      vector_var_vector var_order;
      for (size_t i = 0; i < h * w; ++i)
        var_order.push_back(u.new_vector_variable(depth));
      for (size_t j = 0; j < extra.size(); ++j)
        var_order.push_back(extra[j]);
      var_order.push_back(u.new_vector_variable(NMETADATA));
      return var_order;
    }

} // namespace sill

#include <sill/macros_undef.hpp>
