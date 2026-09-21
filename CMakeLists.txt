cmake_minimum_required(VERSION 3.24)
project(nakshatra LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)

FetchContent_MakeAvailable(json)

set(POSSIBLE_SOURCES
    src/tensor.cpp
    src/ops.cpp
    src/rope.cpp
    src/safetensors.cpp
    src/model.cpp
    src/ir.cpp
    src/pass.cpp
    src/passes.cpp
    src/exercises.cpp
    src/lowering.cpp
    metal/metal_device.mm
)

set(SOURCES "")
foreach(f IN LISTS POSSIBLE_SOURCES)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${f}")
        list(APPEND SOURCES "${f}")
    endif()
endforeach()

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

set_source_files_properties(metal/metal_device.mm PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

add_library(nakshatra_core STATIC ${SOURCES})
target_include_directories(nakshatra_core PUBLIC include)
target_link_libraries(nakshatra_core PUBLIC
    nlohmann_json::nlohmann_json
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    add_executable(nakshatra src/main.cpp)
    target_link_libraries(nakshatra PRIVATE nakshatra_core)
endif()

enable_testing()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_golden.cpp")
  add_executable(test_golden tests/test_golden.cpp)
  target_link_libraries(test_golden PRIVATE nakshatra_core)
  add_test(NAME golden COMMAND test_golden)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_passes.cpp")
  add_executable(test_passes tests/test_passes.cpp)
  target_link_libraries(test_passes PRIVATE nakshatra_core)
  add_test(NAME passes COMMAND test_passes)
endif()
