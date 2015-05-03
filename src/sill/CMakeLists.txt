add_subdirectory(base)

add_prefix(SILL_BASE_SOURCES "base/")
add_library(sill
  ${SILL_BASE_SOURCES}
)
