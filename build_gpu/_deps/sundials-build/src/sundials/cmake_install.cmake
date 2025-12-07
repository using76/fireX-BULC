# Install script for directory: C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/src/sundials

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/hypre")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "C:/msys64/ucrt64/bin/objdump.exe")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  MESSAGE("
Install shared components
")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/ji/Documents/fireX/build_gpu/bin/libsundials_core.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sundials" TYPE FILE FILES
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_adaptcontroller.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_adjointcheckpointscheme.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_adjointstepper.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_band.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_base.hpp"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_context.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_context.hpp"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_convertibleto.hpp"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_core.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_core.hpp"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_dense.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_direct.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_domeigestimator.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_errors.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_futils.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_iterative.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_linearsolver.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_linearsolver.hpp"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_logger.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_math.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_matrix.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_matrix.hpp"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_memory.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_memory.hpp"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_mpi_types.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_nonlinearsolver.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_nonlinearsolver.hpp"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_nvector.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_nvector.hpp"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_profiler.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_profiler.hpp"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_stepper.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_types_deprecated.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_types.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_version.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/sundials_mpi_errors.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sundials/priv" TYPE FILE FILES
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/priv/sundials_context_impl.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/priv/sundials_errors_impl.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/priv/sundials_logger_macros.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sundials/priv" TYPE FILE FILES "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/sundials/priv/sundials_mpi_errors_impl.h")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-build/src/sundials/fmod_int64/cmake_install.cmake")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-build/src/sundials/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
