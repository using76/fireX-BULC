# CMake generated Testfile for 
# Source directory: C:/Users/ji/Documents/fireX/fds-FireX
# Build directory: C:/Users/ji/Documents/fireX/build_gpu
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[FDS Executes]=] "C:/Users/ji/Documents/fireX/build_gpu/fds.exe")
set_tests_properties([=[FDS Executes]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/ji/Documents/fireX/fds-FireX/CMakeLists.txt;327;add_test;C:/Users/ji/Documents/fireX/fds-FireX/CMakeLists.txt;0;")
subdirs("_deps/hypre-build")
subdirs("_deps/sundials-build")
