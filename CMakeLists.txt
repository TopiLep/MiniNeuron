cmake_minimum_required(VERSION 3.25)

project(MiniNeuron
        VERSION 0.1.0
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#
#   Library
#

add_library(MiniNeuron
        src/layer.cpp
        src/network.cpp
        src/initializers.cpp

        includes/layer.h
        includes/network.h
        includes/initializers.h
        includes/activation_types.h
        includes/loss_types.h
        includes/matrix.h
        includes/optimizer_types.h
)

target_include_directories(MiniNeuron
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/includes
)


#
#   Tests
#

include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(MiniNeuronTests
        tests/test_matrix.cpp
        tests/test-network.cpp

)

target_link_libraries(MiniNeuronTests
        PRIVATE
        MiniNeuron
        GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(MiniNeuronTests)

#
#   Demos
#

add_executable(xor_demo
        demos/xor-demo/src/main.cpp
)

target_link_libraries(xor_demo
        PRIVATE
        MiniNeuron
)

add_executable(mnist_demo
        demos/mnist/src/main.cpp
)

target_link_libraries(mnist_demo
        PRIVATE
        MiniNeuron
)


