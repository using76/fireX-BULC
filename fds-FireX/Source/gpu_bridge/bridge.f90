!> \brief GPU Bridge Module for Triton Kernel Integration
!> Provides ISO_C_BINDING interface for Fortran-C-Python communication

MODULE GPU_BRIDGE

USE ISO_C_BINDING
USE PRECISION_PARAMETERS, ONLY: EB, GPU_EB

IMPLICIT NONE (TYPE,EXTERNAL)

PRIVATE

! Public interfaces
PUBLIC :: GPU_INIT, GPU_FINALIZE, GPU_IS_AVAILABLE
PUBLIC :: GPU_RADIATION_COMPUTE, GPU_SYNC
PUBLIC :: RADIATION_GPU_DATA

! GPU availability flag
LOGICAL, SAVE :: GPU_INITIALIZED = .FALSE.
LOGICAL, SAVE :: GPU_AVAILABLE = .FALSE.

!> Radiation data structure for GPU transfer
TYPE, BIND(C) :: RADIATION_GPU_DATA
   TYPE(C_PTR) :: TMP_PTR          !< Temperature field pointer
   TYPE(C_PTR) :: KAPPA_GAS_PTR    !< Gas absorption coefficient pointer
   TYPE(C_PTR) :: IL_PTR           !< Radiation intensity pointer (input/output)
   TYPE(C_PTR) :: QR_PTR           !< Radiation source term pointer (output)
   TYPE(C_PTR) :: EXTCOE_PTR       !< Extinction coefficient pointer
   TYPE(C_PTR) :: SCAEFF_PTR       !< Scattering efficiency pointer
   INTEGER(C_INT) :: IBAR          !< Grid cells in X direction
   INTEGER(C_INT) :: JBAR          !< Grid cells in Y direction
   INTEGER(C_INT) :: KBAR          !< Grid cells in Z direction
   INTEGER(C_INT) :: NRA           !< Number of radiation angles
   INTEGER(C_INT) :: NBAND         !< Number of spectral bands
   REAL(C_FLOAT) :: DX             !< Grid spacing X
   REAL(C_FLOAT) :: DY             !< Grid spacing Y
   REAL(C_FLOAT) :: DZ             !< Grid spacing Z
   REAL(C_FLOAT) :: SIGMA          !< Stefan-Boltzmann constant
END TYPE RADIATION_GPU_DATA

! C function interfaces
INTERFACE
   !> Initialize Python runtime and Triton kernels
   FUNCTION gpu_bridge_init() BIND(C, NAME='gpu_bridge_init')
      IMPORT :: C_INT
      INTEGER(C_INT) :: gpu_bridge_init
   END FUNCTION gpu_bridge_init

   !> Finalize Python runtime
   SUBROUTINE gpu_bridge_finalize() BIND(C, NAME='gpu_bridge_finalize')
   END SUBROUTINE gpu_bridge_finalize

   !> Check if GPU (CUDA) is available
   FUNCTION gpu_bridge_check_gpu() BIND(C, NAME='gpu_bridge_check_gpu')
      IMPORT :: C_INT
      INTEGER(C_INT) :: gpu_bridge_check_gpu
   END FUNCTION gpu_bridge_check_gpu

   !> Compute radiation using Triton kernel
   FUNCTION gpu_radiation_kernel(data) BIND(C, NAME='gpu_radiation_kernel')
      IMPORT :: C_INT, RADIATION_GPU_DATA
      TYPE(RADIATION_GPU_DATA), INTENT(IN) :: data
      INTEGER(C_INT) :: gpu_radiation_kernel
   END FUNCTION gpu_radiation_kernel

   !> Synchronize GPU operations
   SUBROUTINE gpu_bridge_sync() BIND(C, NAME='gpu_bridge_sync')
   END SUBROUTINE gpu_bridge_sync
END INTERFACE

CONTAINS

!> Initialize GPU bridge (Python runtime + Triton)
SUBROUTINE GPU_INIT(STATUS)
   INTEGER, INTENT(OUT) :: STATUS
   INTEGER(C_INT) :: C_STATUS

   IF (GPU_INITIALIZED) THEN
      STATUS = 0
      RETURN
   ENDIF

   ! Initialize Python runtime
   C_STATUS = gpu_bridge_init()
   IF (C_STATUS /= 0) THEN
      STATUS = -1
      GPU_INITIALIZED = .FALSE.
      GPU_AVAILABLE = .FALSE.
      RETURN
   ENDIF

   ! Check GPU availability
   C_STATUS = gpu_bridge_check_gpu()
   GPU_AVAILABLE = (C_STATUS == 1)
   GPU_INITIALIZED = .TRUE.
   STATUS = 0

END SUBROUTINE GPU_INIT

!> Finalize GPU bridge
SUBROUTINE GPU_FINALIZE()
   IF (GPU_INITIALIZED) THEN
      CALL gpu_bridge_finalize()
      GPU_INITIALIZED = .FALSE.
      GPU_AVAILABLE = .FALSE.
   ENDIF
END SUBROUTINE GPU_FINALIZE

!> Check if GPU is available for computation
FUNCTION GPU_IS_AVAILABLE() RESULT(AVAILABLE)
   LOGICAL :: AVAILABLE
   AVAILABLE = GPU_INITIALIZED .AND. GPU_AVAILABLE
END FUNCTION GPU_IS_AVAILABLE

!> Compute radiation heat transfer on GPU
SUBROUTINE GPU_RADIATION_COMPUTE(TMP, KAPPA_GAS, IL, QR, EXTCOE, SCAEFF, &
                                  IBAR, JBAR, KBAR, NRA, NBAND, &
                                  DX, DY, DZ, SIGMA, STATUS)
   ! Input arrays (Fortran native layout)
   REAL(EB), TARGET, INTENT(IN) :: TMP(0:,0:,0:)
   REAL(EB), TARGET, INTENT(IN) :: KAPPA_GAS(:,:,:)
   REAL(EB), TARGET, INTENT(INOUT) :: IL(:,:,:,:)
   REAL(EB), TARGET, INTENT(OUT) :: QR(:,:,:)
   REAL(EB), TARGET, INTENT(IN) :: EXTCOE(:,:,:)
   REAL(EB), TARGET, INTENT(IN) :: SCAEFF(:,:,:)

   ! Grid dimensions
   INTEGER, INTENT(IN) :: IBAR, JBAR, KBAR, NRA, NBAND

   ! Grid spacing and constants
   REAL(EB), INTENT(IN) :: DX, DY, DZ, SIGMA

   ! Output status
   INTEGER, INTENT(OUT) :: STATUS

   ! Local variables
   TYPE(RADIATION_GPU_DATA) :: GPU_DATA
   INTEGER(C_INT) :: C_STATUS

   ! Check if GPU is available
   IF (.NOT. GPU_IS_AVAILABLE()) THEN
      STATUS = -1
      RETURN
   ENDIF

   ! Prepare GPU data structure
   GPU_DATA%TMP_PTR = C_LOC(TMP(0,0,0))
   GPU_DATA%KAPPA_GAS_PTR = C_LOC(KAPPA_GAS(1,1,1))
   GPU_DATA%IL_PTR = C_LOC(IL(1,1,1,1))
   GPU_DATA%QR_PTR = C_LOC(QR(1,1,1))
   GPU_DATA%EXTCOE_PTR = C_LOC(EXTCOE(1,1,1))
   GPU_DATA%SCAEFF_PTR = C_LOC(SCAEFF(1,1,1))

   GPU_DATA%IBAR = INT(IBAR, C_INT)
   GPU_DATA%JBAR = INT(JBAR, C_INT)
   GPU_DATA%KBAR = INT(KBAR, C_INT)
   GPU_DATA%NRA = INT(NRA, C_INT)
   GPU_DATA%NBAND = INT(NBAND, C_INT)

   GPU_DATA%DX = REAL(DX, C_FLOAT)
   GPU_DATA%DY = REAL(DY, C_FLOAT)
   GPU_DATA%DZ = REAL(DZ, C_FLOAT)
   GPU_DATA%SIGMA = REAL(SIGMA, C_FLOAT)

   ! Call Triton kernel via C bridge
   C_STATUS = gpu_radiation_kernel(GPU_DATA)

   IF (C_STATUS /= 0) THEN
      STATUS = -2
      RETURN
   ENDIF

   STATUS = 0

END SUBROUTINE GPU_RADIATION_COMPUTE

!> Synchronize GPU operations
SUBROUTINE GPU_SYNC()
   IF (GPU_INITIALIZED) THEN
      CALL gpu_bridge_sync()
   ENDIF
END SUBROUTINE GPU_SYNC

END MODULE GPU_BRIDGE
