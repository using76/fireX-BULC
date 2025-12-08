# FDS-FireX FP32 HYPRE 호환성 수정 계획

## 문제 진단

### 현재 상태
- **FP64 모드**: 정상 동작
- **FP64+Triton GPU 모드**: 정상 동작
- **FP32 모드**: 런타임 충돌 (0xc0000374 힙 손상)

### 충돌 발생 시점
- "Starting the time-stepping" 직후
- 첫 번째 타임스텝에서 압력 솔버(HYPRE) 호출 시

---

## 근본 원인 분석

### 1. 현재 코드 구조

```
prec.f90:
  USE_FP32 정의 시: EB = SELECTED_REAL_KIND(6) = float (4 bytes)
  기본값:          EB = SELECTED_REAL_KIND(12) = double (8 bytes)

type.f90:1746:
  REAL(EB),ALLOCATABLE, DIMENSION(:) :: F_H, X_H  ! 솔버 RHS/해 벡터

pres.f90:1086:
  REAL(EB),ALLOCATABLE, DIMENSION(:,:) :: D_H_MAT  ! 행렬 값

imkl.f90:343-348:
  SUBROUTINE HYPRE_IJMATRIXSETVALUES(...)
    #ifdef USE_FP32
      REAL(KIND=4), INTENT(IN) :: VALUES(*)  ! FP32
    #else
      REAL(KIND=8), INTENT(IN) :: VALUES(*)  ! FP64
    #endif
```

### 2. 잠재적 문제점

#### A. HYPRE 라이브러리 빌드 설정
CMakeLists.txt:240-243에서 HYPRE_ENABLE_SINGLE 설정:
```cmake
if(USE_FP32)
    set(HYPRE_ENABLE_SINGLE ON CACHE BOOL "" FORCE)
endif()
```

**문제**: HYPRE FetchContent 빌드 시 HYPRE_ENABLE_SINGLE이 올바르게 전달되지 않을 수 있음

#### B. HYPREf.h 헤더 호환성
- 업스트림 HYPRE의 Fortran 헤더는 단정밀도 모드를 완전히 지원하지 않을 수 있음
- `HYPRE_Real` 타입 정의가 C 라이브러리와 불일치

#### C. 함수 이름 맹글링 (Windows)
- gfortran vs ifort 이름 맹글링 차이
- 대소문자 + 언더스코어 규칙 불일치

---

## 수정 계획

### Phase 1: HYPRE 빌드 검증 (필수)

#### 1.1 HYPRE 빌드 설정 확인
```bash
# build_fp32/_deps/hypre-build에서 확인
grep -r "HYPRE_SINGLE" CMakeCache.txt
```

#### 1.2 HYPRE 라이브러리 심볼 검증
```bash
# Windows: dumpbin 또는 nm
nm -C libHYPRE.a | grep HYPRE_IJMatrix
```

#### 1.3 CMakeLists.txt 수정
**파일**: `fds-FireX/CMakeLists.txt`

```cmake
# 변경 전 (라인 239-243):
if(USE_FP32)
    set(HYPRE_ENABLE_SINGLE ON CACHE BOOL "" FORCE)
    message(STATUS "HYPRE single precision enabled for FP32 mode")
endif()

# 변경 후:
if(USE_FP32)
    # HYPRE 단정밀도 활성화 - FetchContent 전에 설정
    set(HYPRE_SINGLE ON CACHE BOOL "" FORCE)
    set(HYPRE_USING_FEI OFF CACHE BOOL "" FORCE)
    message(STATUS "HYPRE single precision (HYPRE_SINGLE) enabled for FP32 mode")
endif()
```

---

### Phase 2: Fortran 인터페이스 수정

#### 2.1 imkl.f90 수정 - 타입 안전성 강화

**파일**: `Source/imkl.f90`

**현재 문제**: HYPRE_REAL 파라미터가 인터페이스에서 직접 사용되지 않음

```fortran
! 현재 코드 (라인 237-241):
#ifdef USE_FP32
   INTEGER, PARAMETER :: HYPRE_REAL = 4
#else
   INTEGER, PARAMETER :: HYPRE_REAL = 8
#endif

! 수정: 명시적 KIND 사용
#ifdef USE_FP32
   INTEGER, PARAMETER :: HYPRE_REAL_KIND = 4
   INTEGER, PARAMETER :: C_HYPRE_REAL = C_FLOAT
#else
   INTEGER, PARAMETER :: HYPRE_REAL_KIND = 8
   INTEGER, PARAMETER :: C_HYPRE_REAL = C_DOUBLE
#endif
```

#### 2.2 ISO_C_BINDING 기반 인터페이스

**수정된 인터페이스 예시**:
```fortran
SUBROUTINE HYPRE_IJVECTORSETVALUES(X, LOCAL_SIZE, ROWS, VALUES, IERR) BIND(C, NAME="HYPRE_IJVectorSetValues")
   USE, INTRINSIC :: ISO_C_BINDING
   INTEGER(C_INT64_T), VALUE, INTENT(IN) :: X
   INTEGER(C_INT), VALUE, INTENT(IN) :: LOCAL_SIZE
   INTEGER(C_INT), INTENT(IN) :: ROWS(*)
#ifdef USE_FP32
   REAL(C_FLOAT), INTENT(IN) :: VALUES(*)
#else
   REAL(C_DOUBLE), INTENT(IN) :: VALUES(*)
#endif
   INTEGER(C_INT), INTENT(OUT) :: IERR
END SUBROUTINE
```

---

### Phase 3: 래퍼 레이어 옵션 (대안)

HYPRE를 직접 수정하지 않고 FDS에서 FP64로 HYPRE를 호출하는 래퍼 레이어:

#### 3.1 pres_hypre_wrapper.f90 (새 파일)

```fortran
MODULE HYPRE_WRAPPER
   USE PRECISION_PARAMETERS
   USE HYPRE_INTERFACE
   IMPLICIT NONE

CONTAINS

#ifdef USE_FP32
   ! FP32 FDS에서 FP64 HYPRE 호출 래퍼
   SUBROUTINE HYPRE_SETVALUES_WRAPPER(VEC, N, ROWS, VALUES_SP, IERR)
      INTEGER(KIND=8), INTENT(IN) :: VEC
      INTEGER, INTENT(IN) :: N
      INTEGER, INTENT(IN) :: ROWS(*)
      REAL(4), INTENT(IN) :: VALUES_SP(*)
      INTEGER, INTENT(OUT) :: IERR

      REAL(8), ALLOCATABLE :: VALUES_DP(:)
      INTEGER :: I

      ALLOCATE(VALUES_DP(N))
      DO I = 1, N
         VALUES_DP(I) = REAL(VALUES_SP(I), 8)
      ENDDO

      CALL HYPRE_IJVECTORSETVALUES_DP(VEC, N, ROWS, VALUES_DP, IERR)
      DEALLOCATE(VALUES_DP)
   END SUBROUTINE
#endif

END MODULE HYPRE_WRAPPER
```

---

### Phase 4: 대안 - HYPRE 비활성화

FP32 모드에서 HYPRE 대신 다른 솔버 사용:

#### 4.1 CMakeLists.txt 수정

```cmake
# FP32 모드에서 HYPRE 비활성화
if(USE_FP32)
    set(USE_HYPRE OFF)
    message(WARNING "HYPRE disabled in FP32 mode - using alternative solver")
endif()
```

#### 4.2 pres.f90 수정

```fortran
#ifdef USE_FP32
   ! FP32 모드: FFT 기반 Poisson 솔버 사용
   ULMAT_SOLVER_LIBRARY = LAPACK_FLAG
#else
   ! FP64 모드: HYPRE PCG 솔버 사용
   ULMAT_SOLVER_LIBRARY = HYPRE_FLAG
#endif
```

---

## 권장 접근 방식

### 우선순위 1: FP64+Triton 모드 사용 (현재 작동)
- FDS는 FP64 (수치 정확도 유지)
- GPU 커널만 FP32 (bridge.c에서 변환)
- **빌드 명령**: `cmake -DUSE_TRITON=ON ..`

### 우선순위 2: Phase 1 + Phase 2 적용
1. HYPRE 빌드 설정 수정
2. imkl.f90 인터페이스 개선
3. 테스트 및 검증

### 우선순위 3: Phase 4 (HYPRE 비활성화)
- 가장 간단하지만 솔버 성능 저하 가능

---

## 수정 파일 목록

| 우선순위 | 파일 | 수정 내용 |
|---------|------|----------|
| 1 | CMakeLists.txt | HYPRE_SINGLE 설정 수정 |
| 2 | Source/imkl.f90 | ISO_C_BINDING 인터페이스 |
| 3 | Source/pres.f90 | (선택) 래퍼 호출 또는 솔버 분기 |

---

## 검증 절차

```bash
# 1. 클린 빌드
rm -rf build_fp32_fixed && mkdir build_fp32_fixed && cd build_fp32_fixed

# 2. FP32 빌드
cmake -DUSE_FP32=ON -DCMAKE_BUILD_TYPE=Debug .. 2>&1 | tee cmake.log
grep -i "hypre\|single\|precision" cmake.log

# 3. HYPRE 빌드 확인
cat _deps/hypre-build/CMakeCache.txt | grep -i single

# 4. 빌드
mingw32-make -j4

# 5. 테스트
mpiexec -n 1 ./fds ../test_simple/simple.fds
```

---

## 최종 권장사항

**현 시점에서 최선의 방법**:

```
FP64 + Triton GPU 모드 사용
- cmake -DUSE_TRITON=ON ..
- HYPRE는 FP64로 정상 동작
- GPU 커널(radiation 등)은 FP32로 가속
- bridge.c가 자동으로 FP64↔FP32 변환 수행
```

이 접근법은:
1. 수치 정확도 유지 (FP64 시뮬레이션)
2. GPU 가속 효과 (FP32 커널)
3. 추가 코드 수정 불필요

FP32 전용 모드가 반드시 필요한 경우에만 위의 수정 계획을 적용하세요.
