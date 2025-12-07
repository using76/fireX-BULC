file(REMOVE_RECURSE
  "../../../../../bin/libsundials_sunlinsoldense.a"
  "../../../../../bin/libsundials_sunlinsoldense.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_sunlinsoldense_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
