
#ifndef SILL_DATASOURCE_INFO_TYPE_HPP
#define SILL_DATASOURCE_INFO_TYPE_HPP

#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/range/forward_range.hpp>
#include <sill/serialization/serialize.hpp>

namespace sill {

  //! Everything you need to know about a datasource.
  struct datasource_info_type {

    finite_var_vector finite_seq;

    vector_var_vector vector_seq;

    std::vector<variable::variable_typenames> var_type_order;

    finite_var_vector finite_class_vars;

    vector_var_vector vector_class_vars;

    datasource_info_type();

    //! Construct a datasource info struct explicitly.
    datasource_info_type
    (const finite_var_vector& finite_seq,
     const vector_var_vector& vector_seq,
     const std::vector<variable::variable_typenames>& var_type_order,
     const finite_var_vector& finite_class_vars,
     const vector_var_vector& vector_class_vars);

    //! Construct a datasource info struct explicitly, without class variables.
    datasource_info_type
    (const forward_range<finite_variable*>& finite_seq,
     const forward_range<vector_variable*>& vector_seq,
     const std::vector<variable::variable_typenames>& var_type_order);

    /**
     * Construct a datasource from a vector of variables.
     * This uses the given ordering, and it sets no class variables.
     */
    explicit datasource_info_type(const forward_range<variable*>& var_seq);

    /**
     * Construct a datasource from a vector of variables.
     * This uses the given ordering, and it sets no class variables.
     */
    explicit
    datasource_info_type(const finite_var_vector& var_seq);

    /**
     * Construct a datasource from a vector of variables.
     * This uses the given ordering, and it sets no class variables.
     */
    explicit
    datasource_info_type(const vector_var_vector& var_seq);

    void save(oarchive& a) const;

    void load(iarchive& a);

    bool operator==(const datasource_info_type& other) const;

    bool operator!=(const datasource_info_type& other) const;

  }; // struct datasource_info_type

} // namespace sill

#endif // #ifndef SILL_DATASOURCE_INFO_TYPE_HPP
