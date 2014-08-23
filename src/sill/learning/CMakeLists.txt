add_subdirectory(crf)
add_subdirectory(dataset_old)
add_subdirectory(discriminative)
add_subdirectory(evaluation)
add_subdirectory(parameter_old)
add_subdirectory(validation)

add_prefix(SILL_CRF_SOURCES "crf/")
add_prefix(SILL_DATASET_SOURCES "dataset_old/")
add_prefix(SILL_EVALUATION_SOURCES "evaluation/")
add_prefix(SILL_DISCRIMINATIVE_SOURCES "discriminative/")
add_prefix(SILL_PARAMETER_SOURCES "parameter_old/")
add_prefix(SILL_VALIDATION_SOURCES "validation/")

set(SILL_LEARNING_SOURCES
  ${SILL_CRF_SOURCES}
  ${SILL_DATASET_SOURCES}
  ${SILL_EVALUATION_SOURCES}
  ${SILL_DISCRIMINATIVE_SOURCES}
  ${SILL_PARAMETER_SOURCES}
  ${SILL_VALIDATION_SOURCES}
  PARENT_SCOPE)
