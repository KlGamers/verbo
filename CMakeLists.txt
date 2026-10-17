cmake_minimum_required(VERSION 3.20)
project(verbo VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(VERBO_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(VERBO_NATIVE "Optimize for the host CPU" ON)
option(VERBO_SHARED "Build the shared library for FFI" ON)
option(VERBO_SANITIZE "Build with ASan and UBSan" OFF)
option(VERBO_SCALAR "Disable SIMD kernels" OFF)
option(VERBO_PYTHON "Install the shared library into the Python package" OFF)

find_package(Threads REQUIRED)

set(VERBO_SRC
  src/json.cpp
  src/unicode.cpp
  src/tokenizer.cpp
  src/pool.cpp
  src/kernels.cpp
  src/attn.cpp
  src/model.cpp
  src/session.cpp
  src/sampler.cpp
  src/capi.cpp)

add_library(verbo_obj OBJECT ${VERBO_SRC})
target_include_directories(verbo_obj PUBLIC include PRIVATE src)
target_compile_options(verbo_obj PRIVATE -fvisibility=hidden -fno-math-errno -Wall -Wextra)
target_link_libraries(verbo_obj PUBLIC Threads::Threads)
if(VERBO_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-mcpu=native HAS_MCPU_NATIVE)
  check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64" AND HAS_MCPU_NATIVE)
    target_compile_options(verbo_obj PUBLIC -mcpu=native)
  elseif(HAS_MARCH_NATIVE)
    target_compile_options(verbo_obj PUBLIC -march=native)
  endif()
endif()
if(VERBO_SCALAR)
  target_compile_definitions(verbo_obj PRIVATE VERBO_FORCE_SCALAR)
endif()
if(VERBO_SANITIZE)
  target_compile_options(verbo_obj PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(verbo_obj PUBLIC -fsanitize=address,undefined)
endif()

add_library(verbo STATIC $<TARGET_OBJECTS:verbo_obj>)
target_include_directories(verbo PUBLIC include)
target_link_libraries(verbo PUBLIC Threads::Threads)
if(VERBO_SANITIZE)
  target_link_options(verbo PUBLIC -fsanitize=address,undefined)
endif()

if(VERBO_SHARED)
  add_library(verbo_shared SHARED $<TARGET_OBJECTS:verbo_obj>)
  set_target_properties(verbo_shared PROPERTIES OUTPUT_NAME verbo)
  target_link_libraries(verbo_shared PRIVATE Threads::Threads)
  if(VERBO_SANITIZE)
    target_link_options(verbo_shared PRIVATE -fsanitize=address,undefined)
  endif()
endif()

add_executable(verbo_cli cli/main.cpp cli/serve.cpp)
set_target_properties(verbo_cli PROPERTIES OUTPUT_NAME verbo)
target_link_libraries(verbo_cli PRIVATE verbo)
target_include_directories(verbo_cli PRIVATE src)

if(VERBO_TESTS)
  include(CTest)
  add_executable(verbo_tests tests/test_core.cpp)
  target_link_libraries(verbo_tests PRIVATE verbo)
  target_include_directories(verbo_tests PRIVATE src)
  add_test(NAME core COMMAND verbo_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  add_executable(tokdump tests/tokdump.cpp)
  target_link_libraries(tokdump PRIVATE verbo)
  target_include_directories(tokdump PRIVATE src)
endif()

if(VERBO_PYTHON AND VERBO_SHARED)
  install(TARGETS verbo_shared LIBRARY DESTINATION verbo RUNTIME DESTINATION verbo)
endif()
