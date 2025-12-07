file(REMOVE_RECURSE
  "../../../../bin/libsundials_idas.a"
  "../../../../bin/libsundials_idas.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_idas_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
