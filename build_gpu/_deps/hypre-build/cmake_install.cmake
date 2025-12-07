# Install script for directory: C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src

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

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/blas/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/lapack/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/utilities/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/multivector/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/krylov/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/seq_mv/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/seq_block_mv/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/parcsr_mv/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/parcsr_block_mv/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/distributed_matrix/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/IJ_mv/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/matrix_matrix/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/distributed_ls/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/parcsr_ls/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/struct_mv/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/struct_ls/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/sstruct_mv/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/sstruct_ls/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/ji/Documents/fireX/build_gpu/lib/libHYPRE.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/HYPRE_config.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/HYPREf.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/HYPRE.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/utilities/HYPRE_utilities.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/utilities/_hypre_utilities.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/utilities/_hypre_utilities.hpp"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/utilities/HYPRE_error_f.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/utilities/fortran.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/utilities/fortran_matrix.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/multivector/csr_matmultivec.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/multivector/interpreter.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/multivector/multivector.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/multivector/par_csr_matmultivec.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/multivector/par_csr_pmvcomm.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/multivector/par_multivector.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/multivector/seq_multivector.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/multivector/temp_multivector.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/krylov/HYPRE_krylov.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/krylov/HYPRE_lobpcg.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/krylov/HYPRE_MatvecFunctions.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/krylov/krylov.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/krylov/lobpcg.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/seq_mv/HYPRE_seq_mv.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/seq_mv/seq_mv.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/seq_block_mv/_hypre_seq_block_mv.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/parcsr_mv/HYPRE_parcsr_mv.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/parcsr_mv/_hypre_parcsr_mv.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/parcsr_block_mv/par_csr_block_matrix.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/parcsr_block_mv/csr_block_matrix.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/distributed_matrix/distributed_matrix.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/distributed_matrix/HYPRE_distributed_matrix_mv.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/distributed_matrix/HYPRE_distributed_matrix_protos.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/distributed_matrix/HYPRE_distributed_matrix_types.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/IJ_mv/aux_parcsr_matrix.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/IJ_mv/aux_par_vector.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/IJ_mv/HYPRE_IJ_mv.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/IJ_mv/_hypre_IJ_mv.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/IJ_mv/IJ_matrix.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/IJ_mv/IJ_vector.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/matrix_matrix/HYPRE_matrix_matrix_protos.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/distributed_ls/pilut/HYPRE_DistributedMatrixPilutSolver_protos.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/distributed_ls/pilut/HYPRE_DistributedMatrixPilutSolver_types.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/parcsr_ls/HYPRE_parcsr_ls.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/parcsr_ls/_hypre_parcsr_ls.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/struct_mv/HYPRE_struct_mv.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/struct_mv/_hypre_struct_mv.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/struct_ls/HYPRE_struct_ls.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/struct_ls/_hypre_struct_ls.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/sstruct_mv/HYPRE_sstruct_mv.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/sstruct_mv/_hypre_sstruct_mv.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/sstruct_ls/HYPRE_sstruct_ls.h"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-src/src/sstruct_ls/_hypre_sstruct_ls.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/HYPRE" TYPE FILE FILES
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/HYPREConfig.cmake"
    "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/HYPREConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/HYPRE/HYPRETargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/HYPRE/HYPRETargets.cmake"
         "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/CMakeFiles/Export/0ad2b8ea5e28c7fd1c74f3fc099d0dee/HYPRETargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/HYPRE/HYPRETargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/HYPRE/HYPRETargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/HYPRE" TYPE FILE FILES "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/CMakeFiles/Export/0ad2b8ea5e28c7fd1c74f3fc099d0dee/HYPRETargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/HYPRE" TYPE FILE FILES "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/CMakeFiles/Export/0ad2b8ea5e28c7fd1c74f3fc099d0dee/HYPRETargets-release.cmake")
  endif()
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/ji/Documents/fireX/build_gpu/_deps/hypre-build/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
