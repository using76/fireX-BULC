file(REMOVE_RECURSE
  "../../../../../../bin/libsundials_fnvecopenmp_mod.a"
  "../../../../../../bin/libsundials_fnvecopenmp_mod.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C Fortran)
  include(CMakeFiles/sundials_fnvecopenmp_mod_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
