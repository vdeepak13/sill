find_package(LibXml++)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_definitions(${LIBXMLPP_DEFINITIONS})
include_directories(${LIBXMLPP_INCLUDE_DIR})
link_libraries(${LIBXMLPP_LIBRARIES})
link_directories(${LIBXMLPP_LINK_DIR})

add_executable(xml_io xml_io.cpp)
add_executable(any_factor_xml any_factor_xml.cpp)
add_executable(decomposable_xml decomposable_xml.cpp)
# add_executable(simple simple.cpp)
