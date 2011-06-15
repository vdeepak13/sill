
#ifndef SILL_RECORD_ITERATOR_HPP
#define SILL_RECORD_ITERATOR_HPP

#include <sill/learning/dataset/record.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forward declarations
  template <typename LA> class dataset;
  template <typename LA> class ds_oracle;
  template <typename LA> class dataset_view;

  /**
   * An iterator over the records of a dataset (in record format).
   *
   * @tparam LA  Linear algebra type.
   */
  template <typename LA>
  class record_iterator
    : public std::iterator<std::forward_iterator_tag, const record<LA> > {

    // Public types
    //==========================================================================
  public:

    typedef LA la_type;

    typedef record<LA>        record_type;
    typedef finite_record     finite_record_type;
    typedef vector_record<LA> vector_record_type;

    typedef typename la_type::matrix_type matrix_type;
    typedef typename la_type::vector_type vector_type;
    typedef typename la_type::value_type  value_type;

    // Public methods
    //==========================================================================

    //! Constructs an iterator which acts as an end iterator.
    record_iterator() : data(NULL), r_valid(false), i(0) { }

    //! Copy constructor.
    record_iterator(const record_iterator& it)
      : data(it.data), r_valid(it.r_valid), i(it.i) {
      r.finite_numbering_ptr = it.r.finite_numbering_ptr;
      r.vector_numbering_ptr = it.r.vector_numbering_ptr;
      if (it.r.fin_own) {
        r.fin_ptr->operator=(*(it.r.fin_ptr));
      } else {
        r.fin_own = false;
        delete(r.fin_ptr);
        r.fin_ptr = it.r.fin_ptr;
      }
      if (it.r.vec_own) {
        r.vec_ptr->operator=(*(it.r.vec_ptr));
      } else {
        r.vec_own = false;
        delete(r.vec_ptr);
        r.vec_ptr = it.r.vec_ptr;
      }
    }

    //! Assignment operator.
    record_iterator& operator=(const record_iterator& rec_it) {
      r.finite_numbering_ptr = rec_it.r.finite_numbering_ptr;
      r.vector_numbering_ptr = rec_it.r.vector_numbering_ptr;
      if (rec_it.r.fin_own) {
        if (r.fin_own) {
          r.fin_ptr->operator=(*(rec_it.r.fin_ptr));
        } else {
          r.fin_own = true;
          r.fin_ptr = new std::vector<size_t>(*(rec_it.r.fin_ptr));
        }
      } else {
        if (r.fin_own) {
          r.fin_own = false;
          delete(r.fin_ptr);
        }
        r.fin_ptr = rec_it.r.fin_ptr;
      }
      if (rec_it.r.vec_own) {
        if (r.vec_own) {
          r.vec_ptr->operator=(*(rec_it.r.vec_ptr));
        } else {
          r.vec_own = true;
          r.vec_ptr = new vector_type(*(rec_it.r.vec_ptr));
        }
      } else {
        if (r.vec_own) {
          r.vec_own = false;
          delete(r.vec_ptr);
        }
        r.vec_ptr = rec_it.r.vec_ptr;
      }
      r_valid = rec_it.r_valid;
      data = rec_it.data;
      i = rec_it.i;
      return *this;
    }

    const record_type& operator*() const {
      load_cur_record();
      return r;
    }

    const record_type* const operator->() const {
      load_cur_record();
      return &r;
    }

    record_iterator& operator++() {
      if (data) {
        ++i;
        r_valid = false;
      }
      return *this;
    }

    record_iterator operator++(int) {
      record_iterator copy(*this);
      if (data) {
        ++i;
        r_valid = false;
      }
      return copy;
    }

    bool operator==(const record_iterator& it) const {
      if (data) {
        if (it.data)
          return i == it.i;
        else
          return i == data->size();
      } else {
        if (it.data)
          return it.i == data->size();
        else
          return true;
      }
    }

    bool operator!=(const record_iterator& it) const {
      return i != it.i;
    }

    //! Returns the weight of the current example, or 0 if the iterator
    //! does not point to an example.
    //! @todo Make this safer!
    value_type weight() const {
      assert(data);
      return data->weight(i);
    }

    //! Resets this record iterator to the first record.
    void reset() {
      i = 0;
      r_valid = false;
    }

    //! Resets this record iterator to record j;
    //! This permits more efficient access to datasets which use records
    //! as native types (than using operator[]).
    void reset(size_t j) {
      assert(data);
      assert(j < data->size());
      i = j;
      r_valid = false;
    }

    // Protected data and methods
    //==========================================================================
  protected:
    friend class dataset<LA>;
    friend class ds_oracle<LA>;
    friend class dataset_view<LA>;

    //! Associated dataset.
    //! WARNING: Other classes should modify this with care!
    const dataset<LA>* data;

    mutable record_type r;

    //! Constructs an iterator which owns its data pointed to record i
    record_iterator(const dataset<LA>* data, size_t i)
      : data(data),
        r(data->finite_numbering_ptr(), data->vector_numbering_ptr(),
          data->vector_dim()),
        r_valid(false), i(i) {
    }

    //! Constructs an iterator which does not own its data pointed to record i
    //! @param fin_ptr  pointer to data
    //! @param vec_ptr  pointer to data
    record_iterator(const dataset<LA>* data, size_t i,
                    std::vector<size_t>* fin_ptr, vector_type* vec_ptr)
      : data(data),
        r(data->finite_numbering_ptr(), data->vector_numbering_ptr(),
          fin_ptr, vec_ptr),
        r_valid(false), i(i) {
    }

    //! Loads the current record if necessary
    void load_cur_record() const {
      if (!r_valid) {
        assert(data);
        assert(i < data->size());
        data->load_record(i, r);
        r_valid = true;
      }
    }

    // Private data
    //==========================================================================
  private:

    //! indicates if the current record is valid
    mutable bool r_valid;

    //! current index into the dataset's records
    size_t i;

  }; // class record_iterator

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_RECORD_ITERATOR_HPP
