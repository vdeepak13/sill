add_executable(dense_table dense_table)
add_executable(mutable_queue mutable_queue.cpp)
add_executable(set_index set_index.cpp)
#add_executable(sparse_table sparse_table.cpp)

add_test(dense_table dense_table)
add_test(mutable_queue mutable_queue)
add_test(set_index set_index)
