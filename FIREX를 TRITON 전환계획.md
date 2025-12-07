

이 문서는 개발팀(혹은 사용자 본인)이 프로젝트를 수행하기 위한 **설계도(Blueprint)** 역할을 합니다.

# ---

**FDS-FireX Triton 가속화 프로젝트 계획서 (RTX 4060 Edition)**

## **1\. 개요 (Executive Summary)**

* **프로젝트 명:** FDS-FireX Triton Acceleration (FP32)  
* **목표:** 소비자용 GPU(RTX 4060)에서 NIST FDS의 실행 속도를 CPU 대비 **10배 이상 가속**.  
* **핵심 전략:**  
  1. **Global FP32:** 하드웨어 성능 제한 해제를 위한 전역 단정밀도(Single Precision) 전환.  
  2. **Hybrid Solver:** 압력해석은 검증된 **HYPRE(GPU)** 유지, 계산 집약적 커널(복사/이류/화학)은 **Triton**으로 대체.  
  3. **Embedded Python:** Fortran 호스트 내에 Python 런타임을 내장하여 오버헤드 없는 커널 호출.

## ---

**2\. 시스템 아키텍처 (System Architecture)**

전체 시스템은 \*\*Host(CPU)\*\*와 \*\*Device(GPU)\*\*가 데이터를 공유하며, **C-Bridge**를 통해 제어권을 교환하는 구조입니다.

| 계층 (Layer) | 구성 요소 (Component) | 역할 (Role) | 구현 언어 |
| :---- | :---- | :---- | :---- |
| **Control** | **FDS Main Loop** | 시뮬레이션 제어, 시간 적분, I/O | Fortran 2018 |
| **Interface** | **C-Bridge** | Fortran 데이터 포인터를 Python 객체로 변환 | C / C++ |
| **Solver A** | **PETSc / HYPRE** | 압력(Poisson) 방정식 풀이 (AMG) | C (Library) |
| **Solver B** | **Triton Controller** | GPU 커널 관리 및 실행 (Launcher) | Python |
| **Kernel** | **Triton Kernels** | **복사, 화학, 이류/확산 연산 수행** | **Triton (Python)** |
| **Hardware** | **RTX 4060** | FP32 연산 및 데이터 저장 (VRAM 8GB) | CUDA / PTX |

## ---

**3\. 데이터 흐름 및 메모리 전략 (Zero-Copy)**

RTX 4060의 성능을 갉아먹는 PCIe 병목을 없애기 위해 \*\*"데이터는 GPU에서 나오지 않는다"\*\*는 원칙을 지킵니다.

1. **초기화 (Init):** 시뮬레이션 시작 시 모든 격자 데이터($U, V, W, T, \\rho$ 등)를 cudaMalloc으로 VRAM에 할당.  
2. **포인터 공유:** FDS(Fortran)는 데이터의 \*\*GPU 메모리 주소(Pointer)\*\*만 가지고 있음.  
3. **Triton 실행:** Fortran이 Python에게 "주소 0x1234에 있는 데이터로 계산해"라고 명령.  
4. **In-Place 연산:** Triton이 VRAM 내에서 데이터를 읽고 쓰고 종료. (CPU로 복사 안 함)  
5. **HYPRE 실행:** HYPRE도 동일한 주소 0x1234를 참조하여 압력 계산.

## ---

**4\. 모듈별 상세 구현 계획 (Implementation Details)**

### **4.1. 사전 준비: FP32 환경 구축 (Foundation)**

가장 먼저 수행해야 할 필수 작업입니다.

* **FDS 소스 수정 (Source/prec.f90):**  
  Fortran  
  \! FP64(8) \-\> FP32(4) 변경  
  INTEGER, PARAMETER :: EB \= SELECTED\_REAL\_KIND(6)

* **PETSc/HYPRE 라이브러리 빌드:**  
  * 옵션: \--with-precision=single \--with-cuda=1 \--download-hypre  
* **Python 환경:**  
  * pip install torch triton cupy-cuda12x

### **4.2. 압력 솔버 (Pressure Solver) \- HYPRE 유지**

Triton으로 재작성하지 않고, 설정값 변경으로 최적화합니다.

* **전략:** FDS FireX가 PETSc를 통해 HYPRE를 호출하도록 설정.  
* **설정 파일 (.petscrc):**  
  Bash  
  \-vec\_type cuda  
  \-mat\_type aijcusparse  
  \-ksp\_type cg  
  \-pc\_type hypre  
  \-pc\_hypre\_type boomeramg  
  \-pc\_hypre\_boomeramg\_strong\_threshold 0.7  \# 3D 문제 최적화

### **4.3. 복사 열전달 (Radiation) \- Triton (최우선 순위)**

RTX 4060의 텐서 코어를 사용하여 가장 큰 가속을 얻는 부분입니다.

* **입력:** 온도($T$), 흡수계수($\\kappa$), 각도 가중치($w$).  
* **알고리즘:** Batched Matrix Multiplication (GEMM).  
* **Triton 커널 구조:**  
  Python  
  @triton.jit  
  def radiation\_kernel(I\_ptr, K\_ptr, Out\_ptr, ...):  
      \# 1\. Block 단위 로드 (Coalesced Access)  
      \# 2\. 각도(Angle) 루프 수행 (Dot Product)  
      \# 3\. 결과 누적 및 저장

### **4.4. 이류 및 화학 (Transport & Chemistry) \- Triton**

기존 FireX의 CUDA 코드를 대체하거나 새로 구현합니다.

* **화학 (Chemistry):**  
  * 각 셀(Cell)이 독립적이므로 **Element-wise** 커널 작성.  
  * 수십만 개 셀을 1차원 벡터로 펴서(Flatten) Triton에 전달.  
* **이류 (Advection):**  
  * **스텐실(Stencil) 연산** ($i, i-1, i+1$ 참조).  
  * Triton의 tl.load 시 오프셋(Offset)을 활용하여 이웃 데이터 로드.  
  * 메모리 경계(Ghost Cell) 처리에 주의.

## ---

**5\. 디렉토리 및 파일 구조 (Directory Structure)**

FDS 소스 코드 내에 Triton 관련 파일을 어떻게 배치할지에 대한 구조도입니다.

Plaintext

fds/  
├── Source/  
│   ├── makefile          \# FP32 라이브러리 링크 수정  
│   ├── prec.f90          \# \[수정\] EB \= 4 (FP32)  
│   ├── radi.f90          \# \[수정\] Triton 호출부 추가  
│   ├── main.f90          \# \[수정\] Python 초기화/종료 호출  
│   │  
│   ├── gpu\_bridge/       \# \[신규\] C-Bridge 폴더  
│   │   ├── bridge.c      \# Fortran \<-\> Python 연결 코드  
│   │   └── bridge.h  
│   │  
│   └── python\_kernels/   \# \[신규\] Triton 커널 폴더  
│       ├── \_\_init\_\_.py  
│       ├── radiation.py  \# 복사 솔버 커널  
│       ├── transport.py  \# 이류/확산 커널  
│       └── chemistry.py  \# 화학 반응 커널  
│  
└── .petscrc              \# \[신규\] HYPRE GPU 설정 파일

## ---

**6\. 단계별 로드맵 (Roadmap)**

### **Phase 1: FP32 전환 및 압력 솔버 가속 (1-2주)**

1. PETSc/HYPRE를 FP32로 빌드.  
2. FDS prec.f90 수정 및 컴파일.  
3. .petscrc 설정 후 기본 예제(FireX) 실행.  
4. **목표:** CPU 대비 압력 솔버 속도 2배 이상, NaN(발산) 없이 실행됨 확인.

### **Phase 2: Python 임베딩 및 브리지 구축 (1주)**

1. bridge.c 작성 (Python 초기화, 포인터 전달 함수).  
2. Fortran main.f90에서 Py\_Initialize() 호출 테스트.  
3. 간단한 "Hello World"를 FDS에서 Python으로 출력 확인.

### **Phase 3: 복사(Radiation) Triton 구현 (2-3주)**

1. radiation.py 작성 (Triton GEMM 커널).  
2. radi.f90의 기존 루프를 주석 처리하고 Bridge 호출로 대체.  
3. RTX 4060에서 단독 모듈 테스트.  
4. **목표:** 복사 계산 시간 50배 단축 확인.

### **Phase 4: 전체 통합 및 이류/화학 확장 (지속)**

1. 나머지 물리 모듈(이류, 화학)도 점진적으로 Triton으로 교체.  
2. 전체 시뮬레이션 검증 (정확도 vs 속도).

## ---

**7\. 예상 리스크 및 대응 (Risk Management)**

| 리스크 | 대응 방안 |
| :---- | :---- |
| **VRAM 부족 (OOM)** | RTX 4060은 8GB뿐임. 격자 크기가 400만 개를 넘으면 cupy나 torch 할당 실패 가능. $\\rightarrow$ **Mesh 분할(MPI) 또는 격자 최적화 필수.** |
| **초기 발산** | FP32 정밀도 문제로 $T=0$에서 압력이 튈 수 있음. $\\rightarrow$ **초기 시간 간격(DT)을 매우 작게($10^{-4}$) 설정하여 시작.** |
| **Python 오버헤드** | 매 스텝 Python 함수 호출 비용. $\\rightarrow$ **가능한 많은 연산을 하나의 커널로 묶어서(Fusing) 호출 횟수 최소화.** |

---

이 문서는 사용자님이 바로 개발에 착수할 수 있도록 **가장 현실적이고 효율적인 경로**를 제시하고 있습니다. \*\*Phase 1(FP32 전환)\*\*부터 시작하십시오.