!> \brief GPU Bridge Module for Triton Integration
!>
!> This module provides the Fortran-C interface for GPU-accelerated computations
!> using Triton kernels. It uses ISO_C_BINDING for interoperability.

MODULE GPU_BRIDGE

USE ISO_C_BINDING
USE PRECISION_PARAMETERS

IMPLICIT NONE (TYPE,EXTERNAL)

!> Python runtime initialization state
LOGICAL, SAVE :: PYTHON_INITIALIZED = .FALSE.

!> GPU acceleration flags
LOGICAL, SAVE :: USE_GPU_RADIATION = .FALSE.
LOGICAL, SAVE :: USE_GPU_TRANSPORT = .FALSE.
LOGICAL, SAVE :: USE_GPU_CHEMISTRY = .FALSE.

!> Radiation data structure for GPU kernel
TYPE, BIND(C) :: RADIATION_GPU_DATA
   TYPE(C_PTR) :: TMP           !< Temperature array pointer
   TYPE(C_PTR) :: KAPPA_GAS     !< Gas absorption coefficient
   TYPE(C_PTR) :: KFST4_GAS     !< Emission term (kappa * 4 * sigma * T^4)
   TYPE(C_PTR) :: IL            !< Radiation intensity
   TYPE(C_PTR) :: UIID          !< Integrated intensity
   TYPE(C_PTR) :: QR            !< Radiative heat source
   TYPE(C_PTR) :: EXTCOE        !< Extinction coefficient
   TYPE(C_PTR) :: SCAEFF        !< Scattering efficiency
   TYPE(C_PTR) :: UIIOLD        !< Previous integrated intensity
   TYPE(C_PTR) :: DX            !< Grid spacing X
   TYPE(C_PTR) :: DY            !< Grid spacing Y
   TYPE(C_PTR) :: DZ            !< Grid spacing Z
   TYPE(C_PTR) :: DLX           !< Direction cosine X
   TYPE(C_PTR) :: DLY           !< Direction cosine Y
   TYPE(C_PTR) :: DLZ           !< Direction cosine Z
   TYPE(C_PTR) :: RSA           !< Solid angle weights
   INTEGER(C_INT) :: IBAR       !< Number of cells in X
   INTEGER(C_INT) :: JBAR       !< Number of cells in Y
   INTEGER(C_INT) :: KBAR       !< Number of cells in Z
   INTEGER(C_INT) :: NRA        !< Number of radiation angles
   INTEGER(C_INT) :: NSB        !< Number of spectral bands
   REAL(C_FLOAT) :: FOUR_SIGMA  !< 4 * Stefan-Boltzmann constant
   REAL(C_FLOAT) :: RFPI        !< 1 / (4 * PI)
   REAL(C_FLOAT) :: RSA_RAT     !< Solid angle ratio
END TYPE RADIATION_GPU_DATA

!> C function interfaces
INTERFACE

   !> Initialize Python runtime and load Triton modules
   FUNCTION init_python_runtime() BIND(C, NAME='init_python_runtime')
      IMPORT :: C_INT
      INTEGER(C_INT) :: init_python_runtime
   END FUNCTION init_python_runtime

   !> Finalize Python runtime
   SUBROUTINE finalize_python_runtime() BIND(C, NAME='finalize_python_runtime')
   END SUBROUTINE finalize_python_runtime

   !> Call Triton radiation kernel
   FUNCTION call_radiation_kernel(data) BIND(C, NAME='call_radiation_kernel')
      IMPORT :: C_INT, RADIATION_GPU_DATA
      TYPE(RADIATION_GPU_DATA), INTENT(IN) :: data
      INTEGER(C_INT) :: call_radiation_kernel
   END FUNCTION call_radiation_kernel

   !> Allocate GPU memory (pinned/unified)
   FUNCTION allocate_gpu_array(size_bytes) BIND(C, NAME='allocate_gpu_array')
      IMPORT :: C_PTR, C_SIZE_T
      INTEGER(C_SIZE_T), VALUE :: size_bytes
      TYPE(C_PTR) :: allocate_gpu_array
   END FUNCTION allocate_gpu_array

   !> Free GPU memory
   SUBROUTINE free_gpu_array(ptr) BIND(C, NAME='free_gpu_array')
      IMPORT :: C_PTR
      TYPE(C_PTR), VALUE :: ptr
   END SUBROUTINE free_gpu_array

   !> Synchronize GPU operations
   SUBROUTINE gpu_sync() BIND(C, NAME='gpu_sync')
   END SUBROUTINE gpu_sync

END INTERFACE

CONTAINS

!> \brief Initialize GPU Bridge and Python runtime
!> \details Called once at simulation startup to initialize Python interpreter
!>          and load Triton kernel modules
SUBROUTINE INITIALIZE_GPU_BRIDGE()

   INTEGER(C_INT) :: status

   IF (.NOT. PYTHON_INITIALIZED) THEN
      status = init_python_runtime()
      IF (status == 0) THEN
         PYTHON_INITIALIZED = .TRUE.
         USE_GPU_RADIATION = .TRUE.
         WRITE(*,'(A)') ' GPU Bridge: Python runtime initialized successfully'
         WRITE(*,'(A)') ' GPU Bridge: Triton kernels loaded'
      ELSE
         WRITE(*,'(A,I0)') ' GPU Bridge: Failed to initialize Python runtime, error code: ', status
         USE_GPU_RADIATION = .FALSE.
      ENDIF
   ENDIF

END SUBROUTINE INITIALIZE_GPU_BRIDGE

!> \brief Finalize GPU Bridge
!> \details Called at simulation end to clean up Python runtime
SUBROUTINE FINALIZE_GPU_BRIDGE()

   IF (PYTHON_INITIALIZED) THEN
      CALL finalize_python_runtime()
      PYTHON_INITIALIZED = .FALSE.
      WRITE(*,'(A)') ' GPU Bridge: Python runtime finalized'
   ENDIF

END SUBROUTINE FINALIZE_GPU_BRIDGE

!> \brief Compute radiation using GPU/Triton kernels
!> \param NM Mesh number
!> \details This subroutine prepares data and calls Triton radiation kernel
SUBROUTINE COMPUTE_RADIATION_GPU(NM)

   INTEGER, INTENT(IN) :: NM
   TYPE(RADIATION_GPU_DATA) :: rad_data
   INTEGER(C_INT) :: status

   ! This is a placeholder - actual implementation requires mesh data access
   ! The full implementation will be in radi.f90 where mesh pointers are available

   WRITE(*,'(A,I0)') ' GPU Bridge: Calling radiation kernel for mesh ', NM

   ! Placeholder for kernel call
   ! status = call_radiation_kernel(rad_data)

END SUBROUTINE COMPUTE_RADIATION_GPU

!> \brief Convert Fortran array to C pointer
!> \param arr Fortran array (assumed contiguous)
!> \return C pointer to array data
FUNCTION FORTRAN_TO_C_PTR_3D(arr) RESULT(ptr)
   REAL(GPU_EB), TARGET, INTENT(IN) :: arr(:,:,:)
   TYPE(C_PTR) :: ptr
   ptr = C_LOC(arr(LBOUND(arr,1), LBOUND(arr,2), LBOUND(arr,3)))
END FUNCTION FORTRAN_TO_C_PTR_3D

!> \brief Convert 1D Fortran array to C pointer
FUNCTION FORTRAN_TO_C_PTR_1D(arr) RESULT(ptr)
   REAL(GPU_EB), TARGET, INTENT(IN) :: arr(:)
   TYPE(C_PTR) :: ptr
   ptr = C_LOC(arr(LBOUND(arr,1)))
END FUNCTION FORTRAN_TO_C_PTR_1D

END MODULE GPU_BRIDGE
