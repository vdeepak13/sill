subdirs(crf dataset dataset2 discriminative object_detection order_based_search parameter structure structure_search)

add_executable(chow_liu chow_liu.cpp)
add_executable(em_mog em_mog.cpp)
add_executable(learn_crf_factor learn_crf_factor.cpp)
add_executable(learn_gaussian_crf_factor learn_gaussian_crf_factor.cpp)

add_test(chow_liu chow_liu)
add_test(em_mog em_mog)
