#include <sill/learning/object_detection/image_oracle.hpp>

#include <sill/macros_def.hpp>

namespace sill {

    // Protected methods
    //==========================================================================

    void image_oracle::init() {
      f_in.open(data_filename.c_str());
      assert(f_in.good());
      vector_numbering_ptr->operator[](metadata_var) = 0;
      current_rec.finite_numbering_ptr->clear();

      line.clear();
      while(f_in.good() && !f_in.eof()) {
        if (f_in.peek() == '\n') {
          getline(f_in, line);
        } else if (f_in.peek() == '|') {
          getline(f_in, line);
        } else if (f_in.peek() == '*') {
          getline(f_in, line);
          size_t equals_index(line.find_first_of('='));
          if (equals_index == std::string::npos) {
            std::cerr << "image_oracle given filename with bad option (no equal"
                      << " sign): " << line << std::endl;
            assert(false);
          }
          if (line.size() >= 13 &&
              line.compare(1,equals_index-1,"fixed_size") == 0) {
            if (line[equals_index+1] == '0')
              fixed_size = false;
            else if (line[equals_index+1] == '1')
              fixed_size = true;
            else
              assert(false);
          }
        } else {
          break;
        }
      }
    }

    // Public methods
    //==========================================================================

    image_oracle& image_oracle::operator=(const image_oracle& o) {
      data_filename = o.data_filename;
      fixed_size = o.fixed_size;
      line = o.line;
      current_rec = o.current_rec;
      u = o.u;
      metadata_var = o.metadata_var;
      vars = o.vars;
      vector_numbering_ptr = o.vector_numbering_ptr;
      f_in.open(data_filename.c_str());
      assert(f_in.good());
      f_in.seekg(o.f_in.tellg());
      return *this;
    }

    bool image_oracle::next() {
      line.clear();
      while (line.size() == 0) {
        if (f_in.bad() || f_in.eof())
          return false;
        getline(f_in, line);
      }
      current_name_ = line;
      while (current_name_.size() > 0)
        if (isspace(current_name_[current_name_.size()-1]))
          current_name_.resize(current_name_.size()-1);
        else
          break;
      getline(f_in, line);
      is.clear();
      is.str(line);
      size_t height = 0, width = 0;
      if (!(is >> height)) {
        std::cerr << "image_oracle read bad line (no image height): "
                  << line << std::endl;
        assert(false);
      }
      if (!(is >> width)) {
        std::cerr << "image_oracle read bad line (no image width): "
                  << line << std::endl;
        assert(false);
      }
      size_t imgsize = height * width;
      assert(imgsize > 0);
      // Set variable numberings, or use the old one.
      if (!fixed_size || vector_numbering_ptr->size() == 0) {
        // Make extra variables if necessary, then create variable numbering.
        for (size_t j = vars.size(); j < imgsize; ++j)
          vars.push_back(u.new_vector_variable(1));
        for (size_t j = vector_numbering_ptr->size() - 1; j < imgsize; ++j)
          vector_numbering_ptr->operator[](vars[j]) = j;
        for (size_t j = vector_numbering_ptr->size() - 2; j >= imgsize; --j)
          vector_numbering_ptr->erase(vars[j]);
        vector_numbering_ptr->operator[](metadata_var) = imgsize;
        current_rec.vector_numbering_ptr = vector_numbering_ptr;
      }
      // Load the image
      vec& vecdata = current_rec.vector();
      vecdata.resize(imgsize + image::NMETADATA);
      for (size_t i = 0; i < imgsize; ++i)
        if (!(is >> vecdata[i]))
          assert(false);
      image::representation(vecdata) = 0;
      image::depth(vecdata) = 1;
      image::true_height(vecdata) = height;
      image::true_width(vecdata) = width;
      image::view_row(vecdata) = 0;
      image::view_col(vecdata) = 0;
      image::view_height(vecdata) = height;
      image::view_width(vecdata) = width;
      image::view_scaleh(vecdata) = 1;
      image::view_scalew(vecdata) = 1;
      return true;
    }

    bool image_oracle::reset() {
      f_in.clear();
      f_in.seekg(0, std::ios::beg);
      while(f_in.good() && !f_in.eof()) {
        switch(f_in.peek()) {
        case '\n':
        case '|':
        case '*':
          getline(f_in, line);
          break;
        default:
          return true;
        }
      }
      return false;
    }

  // Free functions
  //==========================================================================

  boost::shared_ptr<std::vector<image::record_type> >
  load_images(const std::string& filename, universe& u) {
    boost::shared_ptr<std::vector<image::record_type> >
      data_ptr(new std::vector<image::record_type>());
    image_oracle o(filename, u);
    while(o.next())
      data_ptr->push_back(o.current());
    return data_ptr;
  }

} // namespace sill

#include <sill/macros_undef.hpp>
