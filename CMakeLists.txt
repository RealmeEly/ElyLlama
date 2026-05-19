cmake_minimum_required(VERSION 4.3)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif ()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(THIS_PROJECT_NAME "ElyLlama")

option(BUILD_TEST "build test file with ./tests/*.cpp" ON)
option(BACKEND_CPU "Enable CPU backend" ON)

if (NOT BACKEND_CPU)
    message("No backend enabled, enable backend CPU as default.")
endif ()

project(
        ${THIS_PROJECT_NAME}
        LANGUAGES C CXX
        DESCRIPTION "gguf inference tool"
        VERSION 0.1.0
)

# CPU config: avx2，openmp, fma
if (BACKEND_CPU)
    find_package(OpenMP REQUIRED)
    add_definitions(-DBACKEND_CPU)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -mavx2 -mf16c -mfma -fopenmp")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mf16c -mfma -fopenmp")
    endif ()
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

# ------------------------- Necessary Header Files Directories -------------------------
include_directories(
        "${PROJECT_SOURCE_DIR}/include"
)

# Backend header files directories
if (BACKEND_CPU)
    include_directories("${PROJECT_SOURCE_DIR}/include/backend/cpu")
endif ()

# ------------------------- Source Files Collection -------------------------
file(GLOB SRC_FILES
        CMAKE_CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/core/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/model/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/backend/*.cpp"
)

if (BACKEND_CPU)
    file(GLOB_RECURSE CPU_SOURCES
            "${PROJECT_SOURCE_DIR}/src/core/cpu/*.cpp"
            "${PROJECT_SOURCE_DIR}/src/backend/cpu/*.cpp"
    )
endif ()

# ------------------------- Third Party Libraries -------------------------
include_directories("${PROJECT_SOURCE_DIR}/external/nlohmann_json/include")
include_directories("${PROJECT_SOURCE_DIR}/external/cxxopt/include")

# ------------------------- Main Library Build -------------------------
add_library(${PROJECT_NAME} STATIC ${SRC_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
)

if (BACKEND_CPU AND OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif ()

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})


# ------------------------- Installation Config -------------------------
include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
        EXPORT ElyLlamaTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
        DIRECTORY ${PROJECT_SOURCE_DIR}/include/core/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ElyLlama/core
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(
        DIRECTORY ${PROJECT_SOURCE_DIR}/include/model/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ElyLlama/model
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(
        DIRECTORY ${PROJECT_SOURCE_DIR}/include/utils/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ElyLlama/utils
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(
        DIRECTORY ${PROJECT_SOURCE_DIR}/include/backend/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ElyLlama/backend
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

if (BACKEND_CPU)
    install(
            DIRECTORY ${PROJECT_SOURCE_DIR}/include/backend/cpu/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ElyLlama/backend/cpu
            FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
endif ()

install(EXPORT ElyLlamaTargets
        FILE ElyLlamaTargets.cmake
        NAMESPACE ElyLlama::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ElyLlama
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/ElyLlama.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/ElyLlamaConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ElyLlama
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ElyLlamaConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ElyLlama
)


# ------------------------- Test System -------------------------
if (BUILD_TEST)
    enable_testing()
    file(GLOB TEST_FILES "${PROJECT_SOURCE_DIR}/tests/*.cpp")
    foreach (test_file ${TEST_FILES})
        get_filename_component(test_name ${test_file} NAME_WE)
        set(test_name "test_${test_name}")
        add_executable(${test_name} ${test_file})
        target_link_libraries(${test_name} PRIVATE ${PROJECT_NAME})
        target_include_directories(${test_name} PRIVATE ${INCLUDE_DIRS})

        if (BACKEND_CPU)
            target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
        endif ()
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach ()
endif ()