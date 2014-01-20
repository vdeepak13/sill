#ifndef SILL_COMMUTATIVE_SEMIRING_HPP
#define SILL_COMMUTATIVE_SEMIRING_HPP

namespace sill {
  
  /**
   * A base class that represents one of pre-defined commutative semirings
   * on factor types.
   */
  template <typename F>
  struct commutative_semiring {
    //! destructor
    virtual ~commutative_semiring() { }

    //! the "cross" operation (e.g., marginal in the sum-product algorithm)
    virtual F collapse(const F& x, const typename F::domain_type& retain) const = 0;

    //! the "dot" operation (e.g., multiplication in the sum-product algorithm)
    virtual F combine(const F& x, const F& y) const = 0;

    //! an in-place version of the dot operation
    virtual void combine_in(F& x, const F& y) const = 0;

    //! the initial value for the dot operation (e.g., 1 in sum-product)
    virtual F combine_init() const = 0;
  };
  
  //! An object representing the sum product commutative semiring \f$([0,
  //! \infty), +, \times, 0, 1)\f$.
  //! \relates commutative_semiring
  template <typename F>
  struct sum_product : public commutative_semiring<F> {
    F collapse(const F& x, const typename F::domain_type& retain) const {
      return x.marginal(retain);
    }
    F combine(const F& x, const F& y) const {
      return x * y;
    }
    void combine_in(F& x, const F& y) const {
      x *= y;
    }
    F combine_init() const {
      return F(1);
    }
  };

  //! An object representing the max product commutative semiring \f$([0,
  //! \infty), \max, \times, 0, 1)\f$.
  //! \relates commutative_semiring
  template <typename F>
  struct max_product : public commutative_semiring<F> {
    F collapse(const F& x, const typename F::domain_type& retain) const {
      return x.maximum(retain);
    }
    F combine(const F& x, const F& y) const {
      return x * y;
    }
    void combine_in(F& x, const F& y) const {
      x *= y;
    }
    F combine_init() const {
      return F(1);
    }
  };

  //! An object representing the min-sum commutative semiring \f$((-\infty,
  //! \infty], \min, +, \infty, 0)\f$.
  //! \relates commutative_semiring
  template <typename F>
  struct min_sum : public commutative_semiring<F> {
    F collapse(const F& x, const typename F::domain_type& retain) const {
      return x.minimum(retain);
    }
    F combine(const F& x, const F& y) const {
      return x + y;
    }
    void combine_in(F& x, const F& y) const {
      x += y;
    }
    F combine_init() const {
      return F(0);
    }
  };

  //! An object representing the max-sum commutative semiring \f$([-\infty,
  //! \infty), \max, +, -\infty, 0)\f$.
  //! \relates commutative_semiring
  template <typename F>
  struct max_sum : public commutative_semiring<F> {
    F collapse(const F& x, const typename F::domain_type& retain) const {
      return x.maximum(retain);
    }
    F combine(const F& x, const F& y) const {
      return x + y;
    }
    void combine_in(F& x, const F& y) const {
      x += y;
    }
    F combine_init() const {
      return F(0);
    }
  };

  //! An object representing the Boolean commutative semiring \f$(\{0, 1\},
  //! \lor, \land, 0, 1)\f$.
  //! \relates commutative_semiring
  template <typename F>
  struct boolean : public commutative_semiring<F> {
    F collapse(const F& x, const typename F::domain_type& retain) const {
      return x.logical_or(retain);
    }
    F combine(const F& x, const F& y) const {
      return x & y;
    }
    void combine_in(F& x, const F& y) const {
      x &= y;
    }
    F combine_init() const {
      return F(1);
    }
  };

  //! An inplace operation that performs the combine operation of
  //! of a commutative semiring.
  template <typename F>
  struct inplace_combination : public inplace_op<F> {
    //! permits implicit conversions from commutative_semiring
    inplace_combination(const commutative_semiring<F>* csr)
      : csr(csr) { }

    F& operator()(F& x, const F& y) {
      csr->combine_in(x, y);
      return x;
    }

  private:
    const commutative_semiring<F>* csr;
  };
  
} // namespace sill

#endif
