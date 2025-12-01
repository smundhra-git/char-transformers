cmake_minimum_required(VERSION 3.10)
project(CharTransformer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to find OpenMP
if(APPLE)
    set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp;${CMAKE_PREFIX_PATH}")
endif()

find_package(OpenMP)

# Optimization Flags
# Note: Do NOT add -fopenmp here manually; let target_link_libraries handle it.
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
    
    if(NOT OpenMP_CXX_FOUND)
        # If no OpenMP, suppress unknown pragma warnings
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    endif()
endif()

include_directories(src)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(main ${SOURCES})

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
endif()
