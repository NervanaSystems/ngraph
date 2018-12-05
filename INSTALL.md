Tested Platforms:

- Ubuntu 16.04 and 18.04
- CentOS 7.4

Our latest instructions for how to build the library are available 
[in the documentation](https://ngraph.nervanasys.com/docs/latest/buildlb.html).

Use `cmake -LH` after cloning the repo to see the currently-supported 
build options. We recommend using, at the least, something like:  

$ cmake ../ -DCMAKE_INSTALL_PREFIX=~/ngraph_dist -DNGRAPH_USE_PREBUILT_LLVM 
-DNGRAPH_ONNX_IMPORT_ENABLE=ON

