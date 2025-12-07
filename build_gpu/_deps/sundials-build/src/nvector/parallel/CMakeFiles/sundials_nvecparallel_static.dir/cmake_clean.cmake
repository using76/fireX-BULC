file(REMOVE_RECURSE
  "../../../../../bin/libsundials_nvecparallel.a"
  "../../../../../bin/libsundials_nvecparallel.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_nvecparallel_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
