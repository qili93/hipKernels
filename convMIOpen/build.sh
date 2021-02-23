#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Build commands, do not change them
#######################################
build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake .. \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
      -DCMAKE_VERBOSE_MAKEFILE=OFF \
      -DCMAKE_BUILD_TYPE=Debug

make

cd -
echo "ls -l $build_dir"
ls -l $build_dir

./build/main