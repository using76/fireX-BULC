file(REMOVE_RECURSE
  "../../../../../bin/libsundials_sunlinsolband.a"
  "../../../../../bin/libsundials_sunlinsolband.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_sunlinsolband_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
