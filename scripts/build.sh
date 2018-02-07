mkdir build
cd build
cmake .. -DNGRAPH_GPU_ENABLE=TRUE -DNGRAPH_CPU_ENABLE=TRUE -DCUDNN_ROOT_DIR=/usr/lib/x86_64-linux-gnu/ -DCUDNN_INCLUDE_DIR=/usr/include -DZLIB_LIBRARY=/usr/lib/x86_64-linux/gpu/libz.so -DZLIB_INCLUDE_DIR=/usr/include/ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j24 all
