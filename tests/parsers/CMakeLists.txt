add_executable(alchemy alchemy.cpp)
add_executable(bn_parser bn_parser.cpp)
add_executable(decomposable_parser decomposable_parser.cpp)
add_executable(factor_graph_replicate factor_graph_replicate.cpp)
add_executable(uai2alchemy uai2alchemy.cpp)
add_executable(uai2hugin uai2hugin.cpp)
add_executable(uaievid2inst uaievid2inst.cpp)
add_executable(print_model_info print_model_info.cpp)

add_executable(simple_config simple_config.cpp)
add_executable(string_functions string_functions.cpp)

add_test(simple_config simple_config ${CMAKE_CURRENT_LIST_DIR}/test.cfg)
add_test(string_functions string_functions)
