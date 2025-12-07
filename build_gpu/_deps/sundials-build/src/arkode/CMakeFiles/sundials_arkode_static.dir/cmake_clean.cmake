file(REMOVE_RECURSE
  "../../../../bin/libsundials_arkode.a"
  "../../../../bin/libsundials_arkode.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_arkode_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
