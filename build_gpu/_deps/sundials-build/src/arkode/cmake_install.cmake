# Install script for directory: C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/src/arkode

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
Install ARKODE
")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/ji/Documents/fireX/build_gpu/bin/libsundials_arkode.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/arkode" TYPE FILE FILES
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/arkode/arkode.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/arkode/arkode_arkstep.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/arkode/arkode_bandpre.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/arkode/arkode_bbdpre.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/arkode/arkode_butcher.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/arkode/arkode_butcher_dirk.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/arkode/arkode_butcher_erk.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/arkode/arkode_erkstep.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/arkode/arkode_forcingstep.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/arkode/arkode_ls.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/arkode/arkode_lsrkstep.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/arkode/arkode_mristep.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/arkode/arkode_splittingstep.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/arkode/arkode_sprk.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-src/include/arkode/arkode_sprkstep.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-build/src/arkode/fmod_int64/cmake_install.cmake")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/ji/Documents/fireX/build_gpu/_deps/sundials-build/src/arkode/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
