file(REMOVE_RECURSE
  "../../../../../bin/libsundials_fcore_mod.a"
  "../../../../../bin/libsundials_fcore_mod.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C Fortran)
  include(CMakeFiles/sundials_fcore_mod_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
