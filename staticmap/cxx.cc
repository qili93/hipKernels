#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "header.h"

void CXX::Run() {
   Head::GetAll()["CXX1"][1] = 1;
   Head::GetAll()["CXX2"][2] = 2;
}

std::string CXX::GetString() {
    std::ostringstream stream;  
    auto &all_static_vars = Head::GetAll();
    stream << "[CXX] Address of static vars: " << &all_static_vars << std::endl;
    for (auto &val_pair : all_static_vars) {
      auto str = val_pair.first;
      for (auto &int_pair : val_pair.second) {
        stream << "String [" << str << "], Int pair : [" << int_pair.first << ", " << int_pair.second << "]" << std::endl;
      }
    }
    return stream.str();
}
