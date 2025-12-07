file(REMOVE_RECURSE
  "../../../../../bin/libsundials_sunmatrixsparse.a"
  "../../../../../bin/libsundials_sunmatrixsparse.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_sunmatrixsparse_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
