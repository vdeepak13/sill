add_subdirectory(crf)
add_subdirectory(dataset)
add_subdirectory(discriminative)
add_subdirectory(parameter)
add_subdirectory(validation)

add_prefix(SILL_CRF_SOURCES "crf/")
add_prefix(SILL_DATASET_SOURCES "dataset/")
add_prefix(SILL_DISCRIMINATIVE_SOURCES "discriminative/")
add_prefix(SILL_PARAMETER_SOURCES "parameter/")
add_prefix(SILL_VALIDATION_SOURCES "validation/")

set(SILL_LEARNING_SOURCES
  ${SILL_CRF_SOURCES}
  ${SILL_DATASET_SOURCES}
  ${SILL_DISCRIMINATIVE_SOURCES}
  ${SILL_PARAMETER_SOURCES}
  ${SILL_VALIDATION_SOURCES}
  decomposable_parameter_learner
  error_measures
  learn_crf_factor
  PARENT_SCOPE)
