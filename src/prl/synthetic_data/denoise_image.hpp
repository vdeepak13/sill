#ifndef PRL_DENOISE_IMAGE_HPP
#define PRL_DENOISE_IMAGE_HPP

#include <vector>
#include <string>
#include <limits>

#include <boost/random/lagged_fibonacci.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/jpeg_io.hpp>



#include <prl/base/universe.hpp>
#include <prl/model/factor_graph_model.hpp>
#include <prl/base/finite_variable.hpp>


// This must be defined last
#include <prl/macros_def.hpp>

namespace prl {

  // Create the distributions
  void create_distributions(std::vector<double>& mu, 
                            std::vector<double>& var) {
    assert(mu.size() == var.size());
    for(size_t i = 0; i < mu.size(); ++i) {
      mu[i]  = i * 7;
      var[i] = 10*10;
    }
  } // end of create_distributions



  template<typename State>
  void create_belief_image(State& state, 
                           const std::vector<finite_variable*>& variables,
                           const boost::gil::gray32f_view_t& belief_image) {
    for (size_t i = 0; i < variables.size(); ++i) {
      // Decode the variable
      int idx, row, col;
      sscanf(variables[i]->name().c_str(), "[%d](%d,%d)", 
             &idx, &row, &col);
      assert(idx >= 0);
      assert(row >= 0);
      assert(col >= 0);
      // Get the assignment
      typename State::vertex_type vert(variables[i]);
      finite_assignment asg = arg_max( state.belief(vert) );
      // Fill in the pixel in the image
      belief_image(col,row) = asg[variables[i]];
    }
  } // end of create_belief_image


  // Populate the target image
  void create_images(const std::vector<double>& mu, 
                     const std::vector<double>& var,
                     const boost::gil::gray32f_view_t& noisy_image, 
                     const boost::gil::gray32f_view_t& clean_image) {
    // Verify input
    assert(mu.size() == var.size());
//    assert(noisy_image.width() == clean_image.height());
//    assert(noisy_image.height() == clean_image.height());
    for(size_t i = 0; i < var.size(); ++i) assert(var[i] > 0.0);
    
    // Define parameters
    size_t rows = noisy_image.height();
    size_t cols = noisy_image.width();
    size_t arity = mu.size();
    
    // Create sources of randomness
    boost::lagged_fibonacci607 randomness;
    std::vector< boost::normal_distribution<double> > dist(arity);
    for(size_t i = 0; i < arity; ++i) 
      dist[i] = boost::normal_distribution<double>(mu[i], sqrt(var[i]));
    
    // Compute image center
    const double center_r = rows/2.0;
    const double center_c = cols/2.0;
    const double max_radius = std::min(rows, cols) / 2.0;
    
    // Fill out the image
    for(size_t r = 0; r < rows; ++r) {
      for(size_t c = 0; c < cols; ++c) {
        double distance = sqrt((r-center_r)*(r-center_r) + 
                               (c-center_c)*(c-center_c));
        // Compute ring
      //  size_t ring = ((r + c) / 25) % arity;
         size_t ring = 
          static_cast<size_t>(floor(std::min(1.0, distance/max_radius)
                                    * (arity - 1) ) );
        assert(ring >= 0 && ring < mu.size());
          noisy_image(c,r) = dist[ring](randomness);
          clean_image(c,r) = ring;
      }
    }
  } // End of create images


  // Yucheng comment this code
  void save_image(std::string filename, 
                  const boost::gil::gray32f_view_t& in_view) {
    boost::gil::rgb8_image_t out_image(in_view.dimensions());
    boost::gil::rgb8_view_t  out_view(view(out_image));
    size_t rows = in_view.height();
    size_t cols = in_view.width();

    // compute max and min pixel values (because we need to rescale)
    double maxValue = -std::numeric_limits<double>::max();
    double minValue = std::numeric_limits<double>::max();
    for(size_t r = 0; r < rows; ++r) {
      for(size_t c = 0; c < cols; ++c) {
        double ival = in_view(r,c)[0];
        maxValue = std::max(maxValue, ival);
        minValue = std::min(minValue, ival);
      }
    }
    assert(maxValue >= minValue);

    // Write rescaled pixels to the out view
    for(size_t r = 0; r < rows; ++r) {
      for(size_t c = 0; c < cols; ++c) {
        double ival = in_view(r,c)[0];
        // Rescale the pixel value
        if(maxValue == minValue) {
          ival = 0;
        } else {
          ival = (ival - minValue) / (maxValue - minValue) * 255.0;
          ival = std::min(255.0, std::max(0.0, ival));
        }
        // write color in all 3 channels
        out_view(r,c)[0] = static_cast<unsigned char>(ival);
        out_view(r,c)[1] = static_cast<unsigned char>(ival);
        out_view(r,c)[2] = static_cast<unsigned char>(ival);
      }
    }
    jpeg_write_view(filename.c_str(), out_view);
  } // end of save_image
  

  /**
   * variables and factors should be empty and will be filled 
   * in and returned by this fucntion
   */
  template<typename Factor>
  void create_network(universe &u,
                      factor_graph_model<Factor>& fg, 
                      const boost::gil::gray32f_view_t& img, 
                      std::vector<double>& mu, 
                      std::vector<double>& var,
                      double bw,
                      std::vector<finite_variable*>& variables){
    typedef Factor factor_type;
    assert(mu.size() == var.size());
    size_t cardinality = mu.size();
    size_t rows = img.height();
    size_t cols = img.width();
    // Verify that the variances are all positive
    // and create all the variables in the universe
    for(size_t i = 0; i < cardinality; ++i) {
      assert(var[i] > 0.0);
    }
    variables.resize(rows*cols);
    
    // allocate the variables with names that 
    for(size_t i = 0, r = 0; r < rows; ++r) {
      for(size_t c = 0; c < cols; ++c, ++i) {
        std::stringstream strm;
        strm << "[" << i << "](" << r << "," << c << ")";
        strm.flush();
        variables[i] = u.new_finite_variable(strm.str(), cardinality);
      }
    }

    // Compute the factors
    for(size_t r = 0; r < rows; ++r) {
      for(size_t c = 0; c < cols; ++c) {
        // Construct Node factor
        size_t u = r * cols + c;
        finite_domain args;
        args.insert(variables[u]);
        factor_type node_factor(args, 1.0);
        for(size_t asg = 0; asg < cardinality; ++asg){
            node_factor.set_logv(asg,
                              -(img(c,r)[0] - mu[asg])*(img(c,r)[0] - mu[asg]) / 
                              (2.0 * var[asg]));
        }
        // add the final node factor to the factor graph
        fg.add_factor(node_factor);

        // Compute parameters
        double disagreement = -bw;
        double agreement = 0.0;

        // Construct vertical factor
        if( r+1 < rows) {
          size_t v = (r+1) * cols + c;
          finite_domain args = make_domain(variables[u], variables[v]);
          factor_type factor(args, exp(disagreement));
          for(size_t asg = 0; asg < cardinality; ++asg){  
            factor.set_logv(asg,asg,agreement);
          }
          fg.add_factor(factor);
        } // end of vertical factor construction

        // Construct horizontal factor
        if( c+1 < cols) {
          size_t v = r * cols + (c+1);
          finite_domain args = make_domain(variables[u], variables[v]);
          factor_type factor(args, exp(disagreement));
          for(size_t asg = 0; asg < cardinality; ++asg){  
            factor.set_logv(asg,asg,agreement);
          }
          fg.add_factor(factor);
        } // end of horizontal factor construction
      } // end of for c
    } // end of for r
  } // end of create_network
  
} // end of PRL NAMESPACE

#include <prl/macros_undef.hpp>
#endif
