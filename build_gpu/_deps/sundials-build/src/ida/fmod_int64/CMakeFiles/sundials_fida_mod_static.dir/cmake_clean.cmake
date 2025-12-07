file(REMOVE_RECURSE
  "../../../../../bin/libsundials_fida_mod.a"
  "../../../../../bin/libsundials_fida_mod.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C Fortran)
  include(CMakeFiles/sundials_fida_mod_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
