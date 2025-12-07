# ---------------------------------------------------------------
# Programmer(s): Cody J. Balos @ LLNL
# ---------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2025, Lawrence Livermore National Security,
# University of Maryland Baltimore County, and the SUNDIALS contributors.
# Copyright (c) 2013-2025, Lawrence Livermore National Security
# and Southern Methodist University.
# Copyright (c) 2002-2013, Lawrence Livermore National Security.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# ---------------------------------------------------------------


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was SUNDIALSConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)

### ------- Set FOUND status for SUNDIALS components

set(_installed_components "fkinsol_mod;kinsol;fidas_mod;idas;fida_mod;ida;fcvodes_mod;cvodes;fcvode_mod;cvode;farkode_mod;arkode;fsundomeigestpower_mod;sundomeigestpower;fsunnonlinsolfixedpoint_mod;sunnonlinsolfixedpoint;fsunnonlinsolnewton_mod;sunnonlinsolnewton;fsunlinsolsptfqmr_mod;sunlinsolsptfqmr;fsunlinsolspgmr_mod;sunlinsolspgmr;fsunlinsolspfgmr_mod;sunlinsolspfgmr;fsunlinsolspbcgs_mod;sunlinsolspbcgs;fsunlinsolpcg_mod;sunlinsolpcg;fsunlinsoldense_mod;sunlinsoldense;fsunlinsolband_mod;sunlinsolband;fsunmatrixsparse_mod;sunmatrixsparse;fsunmatrixdense_mod;sunmatrixdense;fsunmatrixband_mod;sunmatrixband;fnvecopenmp_mod;nvecopenmp;fnvecmpiplusx_mod;nvecmpiplusx;fnvecparallel_mod;nvecparallel;fnvecmpimanyvector_mod;fnvecmanyvector_mod;nvecmpimanyvector;nvecmanyvector;fnvecserial_mod;nvecserial;fcore_mod;core;")

set(_comp_not_found "")
foreach(_comp ${SUNDIALS_FIND_COMPONENTS})
  if(_comp IN_LIST _installed_components)
    set(SUNDIALS_${_comp}_FOUND TRUE)
  else()
    set(SUNDIALS_${_comp}_FOUND FALSE)
    set(_comp_not_found "${_comp} ${_comp_not_found}")
  endif()
endforeach()

if(_comp_not_found)
  set(SUNDIALS_NOT_FOUND_MESSAGE "Component(s) not found: ${_comp_not_found}")
endif()

### ------- Import SUNDIALS targets

include("${CMAKE_CURRENT_LIST_DIR}/SUNDIALSTargets.cmake")

### ------- Alias targets

set(_SUNDIALS_ALIAS_TARGETS "sundials_fkinsol_mod->sundials_fkinsol_mod_static;sundials_kinsol->sundials_kinsol_static;sundials_fidas_mod->sundials_fidas_mod_static;sundials_idas->sundials_idas_static;sundials_fida_mod->sundials_fida_mod_static;sundials_ida->sundials_ida_static;sundials_fcvodes_mod->sundials_fcvodes_mod_static;sundials_cvodes->sundials_cvodes_static;sundials_fcvode_mod->sundials_fcvode_mod_static;sundials_cvode->sundials_cvode_static;sundials_farkode_mod->sundials_farkode_mod_static;sundials_arkode->sundials_arkode_static;sundials_fsundomeigestpower_mod->sundials_fsundomeigestpower_mod_static;sundials_sundomeigestpower->sundials_sundomeigestpower_static;sundials_fsunnonlinsolfixedpoint_mod->sundials_fsunnonlinsolfixedpoint_mod_static;sundials_sunnonlinsolfixedpoint->sundials_sunnonlinsolfixedpoint_static;sundials_fsunnonlinsolnewton_mod->sundials_fsunnonlinsolnewton_mod_static;sundials_sunnonlinsolnewton->sundials_sunnonlinsolnewton_static;sundials_fsunlinsolsptfqmr_mod->sundials_fsunlinsolsptfqmr_mod_static;sundials_sunlinsolsptfqmr->sundials_sunlinsolsptfqmr_static;sundials_fsunlinsolspgmr_mod->sundials_fsunlinsolspgmr_mod_static;sundials_sunlinsolspgmr->sundials_sunlinsolspgmr_static;sundials_fsunlinsolspfgmr_mod->sundials_fsunlinsolspfgmr_mod_static;sundials_sunlinsolspfgmr->sundials_sunlinsolspfgmr_static;sundials_fsunlinsolspbcgs_mod->sundials_fsunlinsolspbcgs_mod_static;sundials_sunlinsolspbcgs->sundials_sunlinsolspbcgs_static;sundials_fsunlinsolpcg_mod->sundials_fsunlinsolpcg_mod_static;sundials_sunlinsolpcg->sundials_sunlinsolpcg_static;sundials_fsunlinsoldense_mod->sundials_fsunlinsoldense_mod_static;sundials_sunlinsoldense->sundials_sunlinsoldense_static;sundials_fsunlinsolband_mod->sundials_fsunlinsolband_mod_static;sundials_sunlinsolband->sundials_sunlinsolband_static;sundials_fsunmatrixsparse_mod->sundials_fsunmatrixsparse_mod_static;sundials_sunmatrixsparse->sundials_sunmatrixsparse_static;sundials_fsunmatrixdense_mod->sundials_fsunmatrixdense_mod_static;sundials_sunmatrixdense->sundials_sunmatrixdense_static;sundials_fsunmatrixband_mod->sundials_fsunmatrixband_mod_static;sundials_sunmatrixband->sundials_sunmatrixband_static;sundials_fnvecopenmp_mod->sundials_fnvecopenmp_mod_static;sundials_nvecopenmp->sundials_nvecopenmp_static;sundials_fnvecmpiplusx_mod->sundials_fnvecmpiplusx_mod_static;sundials_nvecmpiplusx->sundials_nvecmpiplusx_static;sundials_fnvecparallel_mod->sundials_fnvecparallel_mod_static;sundials_nvecparallel->sundials_nvecparallel_static;sundials_fnvecmpimanyvector_mod->sundials_fnvecmpimanyvector_mod_static;sundials_fnvecmanyvector_mod->sundials_fnvecmanyvector_mod_static;sundials_nvecmpimanyvector->sundials_nvecmpimanyvector_static;sundials_nvecmanyvector->sundials_nvecmanyvector_static;sundials_fnvecserial_mod->sundials_fnvecserial_mod_static;sundials_nvecserial->sundials_nvecserial_static;sundials_fcore_mod->sundials_fcore_mod_static;sundials_core->sundials_core_static;")
foreach(ptr ${_SUNDIALS_ALIAS_TARGETS})
  string(REGEX REPLACE "sundials_" "" ptr "${ptr}")
  string(REGEX MATCHALL "([A-Za-z_]+)->([A-Za-z_]+)"
         _matches "${ptr}")
  set(_pointer ${CMAKE_MATCH_1})
  set(_pointee ${CMAKE_MATCH_2})
  if(NOT TARGET SUNDIALS::${_pointer})
    add_library(SUNDIALS::${_pointer} INTERFACE IMPORTED)
    target_link_libraries(SUNDIALS::${_pointer} INTERFACE SUNDIALS::${_pointee})
  endif()
endforeach()

### ------- Create TPL imported targets

if("ON" AND NOT TARGET MPI::MPI_C)
  set(MPI_C_COMPILER "C:/msys64/ucrt64/bin/mpicc.exe")
  find_dependency(MPI)
endif()

if("ON" AND NOT TARGET OpenMP::OpenMP_C)
  find_dependency(OpenMP)
endif()

if("" AND NOT TARGET caliper)
  find_dependency(CALIPER PATHS "")
endif()

if("" AND NOT TARGET adiak::adiak)
  find_dependency(adiak PATHS "")
endif()

if("OFF" AND NOT (TARGET CUDA::cudart AND TARGET CUDA::cublas
   AND TARGET CUDA::cusparse AND TARGET CUDA::cusolver))
  find_dependency(CUDAToolkit)
endif()

if("OFF" AND NOT TARGET Ginkgo::ginkgo)
  if(NOT TARGET hwloc AND NOT HWLOC_DIR)
    set(HWLOC_DIR "")
  endif()
  find_dependency(Ginkgo PATHS "")
endif()

if("OFF" AND NOT TARGET SUNDIALS::HYPRE)
  add_library(SUNDIALS::HYPRE INTERFACE IMPORTED)
  target_link_libraries(SUNDIALS::HYPRE INTERFACE "")
  set_target_properties(SUNDIALS::HYPRE PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
endif()

if("OFF" AND NOT TARGET SUNDIALS::KLU)
  if("")
    find_dependency(KLU)
  else()
    add_library(SUNDIALS::KLU INTERFACE IMPORTED)
    target_link_libraries(SUNDIALS::KLU INTERFACE "")
    set_target_properties(SUNDIALS::KLU PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
  endif()
endif()

if("OFF" AND NOT TARGET Kokkos::kokkos)
  find_dependency(Kokkos PATHS "")
endif()

if("OFF" AND NOT TARGET Kokkos::kokkoskernels)
  find_dependency(KokkosKernels PATHS "")
endif()

if("OFF" AND NOT TARGET LAPACK::LAPACK)
  # For some reason find_dependency does not find the libraries if the variables
  # below are internal rather than CACHE variables
  set(BLAS_LIBRARIES "" CACHE "FILEPATH" "BLAS libraries")
  set(BLAS_LINKER_FLAGS "" CACHE "STRING" "BLAS linker flags")
  set(LAPACK_LIBRARIES "" CACHE "FILEPATH" "LAPACK libraries")
  set(LAPACK_LINKER_FLAGS "" CACHE "STRING" "LAPACK linker flags")
  find_dependency(LAPACK)
endif()

if("OFF" AND NOT TARGET SUNDIALS::PETSC)
  add_library(SUNDIALS::PETSC INTERFACE IMPORTED)
  target_link_libraries(SUNDIALS::PETSC INTERFACE "")
  set_target_properties(SUNDIALS::PETSC PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")

  # for backwards compatibility
  foreach(suffix SYS VEC MAT DM KSP SNES TS ALL)
    if(NOT TARGET SUNDIALS::PETSC_${suffix})
      add_library(SUNDIALS::PETSC_${suffix} INTERFACE IMPORTED)
      set_target_properties (SUNDIALS::PETSC_${suffix} PROPERTIES
        INTERFACE_LINK_LIBRARIES "SUNDIALS::PETSC")
    endif()
  endforeach()

  if("" MATCHES "Kokkos::kokkos")
    if(NOT TARGET Kokkos::kokkoskernels)
      find_dependency(KokkosKernels PATHS "")
    endif()
    if(NOT TARGET Kokkos::kokkos)
      find_dependency(Kokkos PATHS "")
    endif()
  endif()
endif()

if("OFF" AND NOT TARGET SUNDIALS::MAGMA)
  add_library(SUNDIALS::MAGMA INTERFACE IMPORTED)
  target_link_libraries(SUNDIALS::MAGMA INTERFACE "")
  set_target_properties(SUNDIALS::MAGMA PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
endif()

if("OFF" AND NOT TARGET MKL)
  find_dependency(MKL PATHS "")
endif()

if("OFF" AND NOT TARGET SUNDIALS::SUPERLUDIST)
  add_library(SUNDIALS::SUPERLUDIST INTERFACE IMPORTED)
  target_link_libraries(SUNDIALS::SUPERLUDIST INTERFACE "")
  set_target_properties(SUNDIALS::SUPERLUDIST PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
endif()

if("OFF" AND NOT TARGET SUNDIALS::SUPERLUMT)
  add_library(SUNDIALS::SUPERLUMT INTERFACE IMPORTED)
  target_link_libraries(SUNDIALS::SUPERLUMT INTERFACE "")
  set_target_properties(SUNDIALS::SUPERLUMT PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
endif()

if("OFF" AND NOT TARGET RAJA)
  if(NOT TARGET camp AND NOT camp_DIR)
    set(camp_DIR "")
  endif()
  find_dependency(RAJA PATHS "")
endif()

if("OFF" AND NOT TARGET Tpetra::all_libs)
  find_dependency(Trilinos COMPONENTS Tpetra PATHS "")
endif()

if("OFF" AND NOT TARGET SUNDIALS::XBRAID)
  add_library(SUNDIALS::XBRAID INTERFACE IMPORTED)
  target_link_libraries(SUNDIALS::XBRAID INTERFACE "")
  set_target_properties(SUNDIALS::XBRAID PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
endif()

### ------- Check if required components were found

check_required_components(SUNDIALS)
