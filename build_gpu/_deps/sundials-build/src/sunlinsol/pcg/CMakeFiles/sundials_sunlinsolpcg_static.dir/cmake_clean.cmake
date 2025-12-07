file(REMOVE_RECURSE
  "../../../../../bin/libsundials_sunlinsolpcg.a"
  "../../../../../bin/libsundials_sunlinsolpcg.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_sunlinsolpcg_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
