cmake_minimum_required(VERSION 3.23)
project(Lab3)

set(CMAKE_CXX_STANDARD 14)

find_package(MPI) #make it REQUIRED
if (MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    target_link_libraries(my_mpi_target ${MPI_C_LIBRARIES})
    find_package(OpenMP)
endif()

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(Lab3 main.cpp)
