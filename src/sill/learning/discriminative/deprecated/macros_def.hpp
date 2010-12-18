
// Defines macros for creating create() and clone() functions for algorithms
// in learning/discriminative/

// Generate create() functions for binary meta-learner with Objective template
//  parameter
#define META_CREATE_FUNCTIONS(meta_name)                                  \
    boost::shared_ptr<binary_classifier>                     \
    createB(statistics& stats) const {                       \
      boost::shared_ptr<binary_classifier>                   \
        bptr(new meta_name<Objective>                         \
             (stats, wl_ptr, this->params));                                  \
      return bptr;                                                        \
    }                                                                     \
    boost::shared_ptr<multiclass_classifier>                 \
    createMC(statistics& stats) const {                      \
      boost::shared_ptr<multiclass_classifier>               \
        bptr(new meta_name<Objective>                         \
             (stats, wl_ptr, this->params));                                  \
      return bptr;                                                        \
    }                                                                     \

// Generate create() functions for binary meta-learner without Objective
#define META_CREATE_FUNCTIONS2(meta_name)                                  \
    boost::shared_ptr<binary_classifier>                     \
    createB(statistics& stats) const {                       \
      boost::shared_ptr<binary_classifier>                   \
        bptr(new meta_name                         \
             (stats, wl_ptr, this->params));                                  \
      return bptr;                                                        \
    }                                                                     \
    boost::shared_ptr<multiclass_classifier>                 \
    createMC(statistics& stats) const {                      \
      boost::shared_ptr<multiclass_classifier>               \
        bptr(new meta_name                         \
             (stats, wl_ptr, this->params));                                  \
      return bptr;                                                        \
    }                                                                     \

// Generate create() functions for multiclass meta-learner with Objective
//  template parameter
#define META_CREATE_FUNCTIONS_MULTI(meta_name)                            \
    boost::shared_ptr<multiclass_classifier>                 \
    createMC(statistics& stats) const {                      \
      boost::shared_ptr<multiclass_classifier>               \
        bptr(new meta_name<Objective>                         \
             (stats, wl_ptr, this->params));                                  \
      return bptr;                                                        \
    }                                                                     \

// Generate create() functions for multiclass meta-learner with no Objective
//  template parameter
#define META_CREATE_FUNCTIONS_MULTI2(meta_name)                           \
    boost::shared_ptr<multiclass_classifier>                 \
    createMC(statistics& stats) const {                      \
      boost::shared_ptr<multiclass_classifier>               \
        bptr(new meta_name                                   \
             (stats, wl_ptr, this->params));                                  \
      return bptr;                                                        \
    }                                                                     \

// Generate create() functions for binary base learner with Objective template
//  parameter
#define BASE_CREATE_FUNCTIONS(base_name)                                  \
    boost::shared_ptr<binary_classifier>                     \
    createB(statistics& stats) const {                       \
      boost::shared_ptr<binary_classifier>                   \
        bptr(new base_name<Objective>                         \
             (stats, this->params));                                      \
      return bptr;                                                        \
    }                                                                     \
    boost::shared_ptr<multiclass_classifier>                 \
    createMC(statistics& stats) const {                      \
      boost::shared_ptr<multiclass_classifier>               \
        bptr(new base_name<Objective>                         \
             (stats, this->params));                                      \
      return bptr;                                                        \
    }                                                                     \

// Generate create() functions for binary base learner with no Objective
//  template parameter
#define BASE_CREATE_FUNCTIONS2(base_name)                                 \
    boost::shared_ptr<binary_classifier>                     \
    createB(statistics& stats) const {                       \
      boost::shared_ptr<binary_classifier>                   \
        bptr(new base_name                                   \
             (stats, this->params));                                      \
      return bptr;                                                        \
    }                                                                     \
    boost::shared_ptr<multiclass_classifier>                 \
    createMC(statistics& stats) const {                      \
      boost::shared_ptr<multiclass_classifier>               \
        bptr(new base_name                                   \
             (stats, this->params));                                      \
      return bptr;                                                        \
    }
