add_subdirectory(crf)
add_subdirectory(dataset)
add_subdirectory(discriminative)
add_subdirectory(object_detection)
add_subdirectory(parameter)

add_prefix(PRL_CRF_SOURCES "crf/")
add_prefix(PRL_DATASET_SOURCES "dataset/")
add_prefix(PRL_DISCRIMINATIVE_SOURCES "discriminative/")
add_prefix(PRL_OBJECT_DETECTION_SOURCES "object_detection/")
add_prefix(PRL_PARAMETER_SOURCES "parameter/")

set(PRL_LEARNING_SOURCES
  ${PRL_CRF_SOURCES}
  ${PRL_DATASET_SOURCES}
  ${PRL_DISCRIMINATIVE_SOURCES}
  ${PRL_OBJECT_DETECTION_SOURCES}
  ${PRL_PARAMETER_SOURCES}
  crossval_builder
  decomposable_parameter_learner
  em_mog
  error_measures
  #learn_crf_factor
  learn_factor
  parameter_grid
  PARENT_SCOPE)
