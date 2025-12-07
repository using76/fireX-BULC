file(REMOVE_RECURSE
  "../../../../bin/libsundials_kinsol.a"
  "../../../../bin/libsundials_kinsol.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_kinsol_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
