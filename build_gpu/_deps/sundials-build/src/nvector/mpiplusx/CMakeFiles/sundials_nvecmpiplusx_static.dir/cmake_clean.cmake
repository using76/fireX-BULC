file(REMOVE_RECURSE
  "../../../../../bin/libsundials_nvecmpiplusx.a"
  "../../../../../bin/libsundials_nvecmpiplusx.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_nvecmpiplusx_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
