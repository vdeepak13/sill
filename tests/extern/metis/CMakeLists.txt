link_libraries( metis )

add_executable(test_metis test_metis.cpp)
target_link_libraries(test_metis metis)
