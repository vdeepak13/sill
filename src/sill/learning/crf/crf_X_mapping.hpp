#ifndef SILL_CRF_X_MAPPING_HPP
#define SILL_CRF_X_MAPPING_HPP

#include <set>

#include <sill/base/stl_util.hpp>
#include <sill/factor/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class passed as a parameter to crf_X_mapping to specify what inputs X
   * to use for each factor involving some subset of Y.
   * This supports:
   *  - (anything) --> X
   *    Return the same X for all cliques in Y.
   *  - Y_C --> X_C
   *    For a given clique in Y, look up a subset of X in a map.
   *  - Y_C --> union_{i \in C} X_i
   *    For a given clique in Y, look up subsets of X for each Y_i in
   *    the clique, and return the union of the subsets.
   *
   * @tparam FactorType  Type which fits the CRFfactor concept.
   */
  template <typename FactorType>
  class crf_X_mapping {

    concept_assert((sill::CRFfactor<FactorType>));

    //! CRF factor type
    typedef FactorType crf_factor;

    //! Type of output variable Y.
    typedef typename crf_factor::output_variable_type output_variable_type;

    //! Type of input variable X.
    typedef typename crf_factor::input_variable_type input_variable_type;

    //! Type of domain for variables in Y.
    typedef typename crf_factor::output_domain_type output_domain_type;

    //! Type of domain for variables in X.
    typedef typename crf_factor::input_domain_type input_domain_type;

    //! Indicates which method to use for mapping Y cliques to inputs from X:
    //!  - 0: Use X_map.
    //!  - 1: Use all_X for each clique in Y.
    //!  - 2: Use X_map2.
    size_t X_map_type_;

    //! (If X_map_type_ == 0)
    //! Map from sets of variables in Y to sets of variables in X to use
    //! as local evidence.
    std::map<output_domain_type, copy_ptr<input_domain_type> > X_map_;

    //! (If X_map_type_ == 1)
    //! All X variables.
    copy_ptr<input_domain_type> all_X_;

    //! (If X_map_type_ == 2)
    //! Map from single variables in Y to sets of inputs in X.
    //! The inputs for sets of Y are the unions of the inputs for each Y.
    std::map<output_variable_type*, copy_ptr<input_domain_type> > X_map2_;

    // Public methods: Constructors
    //==========================================================================
  public:
    //! Default constructor which provides an empty set X for all domains Y.
    crf_X_mapping() : X_map_type_(0) { }

    /**
     * Constructor for mappings:
     *  (set of Y vars) --> (set of X vars)
     * Note: If you allow factors with domains of up to size k,
     *       then X_map must specify any X variables
     *       you want to appear in factors of size 1 to size k.
     * @param X_map  Map from sets of variables in Yvars to sets of
     *               variables in Xvars to use as local evidence.
     */
    crf_X_mapping
    (const std::map<output_domain_type, copy_ptr<input_domain_type> >& X_map_)
      : X_map_type_(0), X_map_(X_map_) { }

    /**
     * Constructor for mappings:
     *  (set of Y vars) --> (all X vars)
     * Note: This uses all X variables for each factor.  If there are many X
     *       variables, you should use an L1-regularized regressor.
     * @param all_X_  All X variables
     */
    crf_X_mapping(copy_ptr<input_domain_type> all_X_)
      : X_map_type_(1), all_X_(all_X_) { }

    /**
     * Constructor for mappings:
     *  (set of Y vars) --> (all X vars)
     * Note: This uses all X variables for each factor.  If there are many X
     *       variables, you should use an L1-regularized regressor.
     * @param all_X_  All X variables
     */
    crf_X_mapping(const input_domain_type& all_X_)
      : X_map_type_(1), all_X_(new input_domain_type(all_X_)) { }

    /**
     * Constructor for mappings:
     *  (single Y var) --> (set of X vars)
     * Note: This constructor lets the user specify
     *       a set of X variables for each Y variable.
     *       This algorithm chooses the domain of a factor f by using the
     *       union of the X variable sets of the Y variables in f.
     */
    crf_X_mapping
    (const std::map<output_variable_type*,copy_ptr<input_domain_type> >&
     X_map2_)
      : X_map_type_(2), X_map2_(X_map2_) { }

    //! Serialize members
    void save(oarchive & ar) const {
      ar << X_map_type_;
      switch (X_map_type_) {
      case 0:
        {
          ar << X_map_.size();
          typename
            std::map<output_domain_type, copy_ptr<input_domain_type> >
            ::const_iterator it = X_map_.begin();
          while (it != X_map_.end()) {
            ar << it->first << *(it->second);
            ++it;
          }
        }
        break;
      case 1:
        ar << *all_X_;
        break;
      case 2:
        {
          ar << X_map2_.size();
          typename
            std::map<output_variable_type*, copy_ptr<input_domain_type> >
            ::const_iterator it = X_map2_.begin();
          while (it != X_map2_.end()) {
            ar << it->first << *(it->second);
            ++it;
          }
        }
        break;
      default:
        throw std::runtime_error
          ("crf_X_mapping::save called on instance with invalid X_map_type_");
      }
    } // save

    //! Deserialize members
    void load(iarchive & ar) {
      X_map_.clear();
      all_X_.reset();
      X_map2_.clear();

      ar >> X_map_type_;
      switch (X_map_type_) {
      case 0:
        {
          size_t mapsize;
          ar >> mapsize;
          output_domain_type outdom;
          input_domain_type indom;
          for (size_t i = 0; i < mapsize; ++i) {
            ar >> outdom >> indom;
            X_map_[outdom] =
              copy_ptr<input_domain_type>(new input_domain_type(indom));
          }
        }
        break;
      case 1:
        all_X_.reset(new input_domain_type());
        ar >> *all_X_;
        break;
      case 2:
        {
          size_t mapsize;
          ar >> mapsize;
          output_variable_type* outvar;
          input_domain_type indom;
          for (size_t i = 0; i < mapsize; ++i) {
            ar >> outvar >> indom;
            X_map2_[outvar] =
              copy_ptr<input_domain_type>(new input_domain_type(indom));
          }
        }
        break;
      default:
        throw std::runtime_error
          ("crf_X_mapping::load read invalid X_map_type_");
      }
    } // load

    // Public methods: Getters
    //==========================================================================

    //! Returns the X variables for the given Y variables.
    copy_ptr<input_domain_type>
    operator[](const output_domain_type& Y) const {
      switch(X_map_type_) {
      case 0:
        {
          typename std::map<output_domain_type, copy_ptr<input_domain_type> >::const_iterator
            it(X_map_.find(Y));
          if (it == X_map_.end())
            return copy_ptr<input_domain_type>(new input_domain_type());
          else
            return it->second;
        }
      case 1:
        return all_X_;
      case 2:
        if (Y.size() == 1) {
          typename output_domain_type::const_iterator Y_it(Y.begin());
          typename std::map<output_variable_type*, copy_ptr<input_domain_type> >::const_iterator
            it(X_map2_.find(*Y_it));
          if (it == X_map2_.end())
            return copy_ptr<input_domain_type>(new input_domain_type());
          else
            return it->second;
        } else if (Y.size() == 0) {
          return copy_ptr<input_domain_type>(new input_domain_type());
        } else {
          copy_ptr<input_domain_type> mappedX(new input_domain_type());
          foreach(output_variable_type* fv, Y) {
            typename std::map<output_variable_type*, copy_ptr<input_domain_type> >::const_iterator
              it(X_map2_.find(fv));
            if (it == X_map2_.end())
              continue;
            else
              mappedX->insert(it->second->begin(), it->second->end());
          }
          return mappedX;
        }
      default:
        assert(false);
      }
    } // end of function operator[]()

    //! Indicates which method to use for mapping Y cliques to inputs from X:
    //!  - 0: Use X_map.
    //!  - 1: Use all_X for each clique in Y.
    //!  - 2: Use X_map2.
    size_t X_map_type() const {
      return X_map_type_;
    }

    //! (If X_map_type == 0)
    //! Map from sets of variables in Y to sets of variables in X to use
    //! as local evidence.
    const std::map<output_domain_type, copy_ptr<input_domain_type> >&
    X_map() const {
      return X_map_;
    }

    //! (If X_map_type == 1)
    //! All X variables.
    copy_ptr<input_domain_type> all_X() const {
      return all_X_;
    }

    //! (If X_map_type == 2)
    //! Map from single variables in Y to sets of inputs in X.
    //! The inputs for sets of Y are the unions of the inputs for each Y.
    const std::map<output_variable_type*, copy_ptr<input_domain_type> >&
    X_map2() const {
      return X_map2_;
    }

  }; // class crf_X_mapping

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_CRF_X_MAPPING_HPP
