file(REMOVE_RECURSE
  "../../../../../bin/libsundials_sunmatrixdense.a"
  "../../../../../bin/libsundials_sunmatrixdense.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_sunmatrixdense_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
