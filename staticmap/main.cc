#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <unordered_map>

#include "header.h"

int main(int argc, char **argv) {
    CXX obj_cxx;
    Clang obj_clang;
    HIPCC obj_hipcc;

    obj_cxx.Run();
    obj_clang.Run();
    obj_hipcc.Run();
    
    std::cout << obj_cxx.GetString() << std::endl;
    std::cout << obj_clang.GetString() << std::endl;
    std::cout << obj_hipcc.GetString() << std::endl;
    
    return 0;
}