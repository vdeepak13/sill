subdirs(crf dataset dataset3 discriminative parameter structure structure_search)

add_executable(chow_liu chow_liu.cpp)
add_executable(learn_crf_factor learn_crf_factor.cpp)
add_executable(learn_gaussian_crf_factor learn_gaussian_crf_factor.cpp)

add_test(chow_liu chow_liu)
