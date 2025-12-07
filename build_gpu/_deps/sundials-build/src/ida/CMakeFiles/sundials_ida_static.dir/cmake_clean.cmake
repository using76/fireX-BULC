file(REMOVE_RECURSE
  "../../../../bin/libsundials_ida.a"
  "../../../../bin/libsundials_ida.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_ida_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
