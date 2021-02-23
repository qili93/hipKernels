#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "header.h"

void Clang::Run() {
   Head::GetAll()["Clang1"][1] = 1;
   Head::GetAll()["Clang2"][2] = 2;
}

std::string Clang::GetString() {
   std::ostringstream stream;  
   auto &all_static_vars = Head::GetAll();
   stream << "[Clang] Address of static vars: " << &all_static_vars << std::endl;
   for (auto &val_pair : all_static_vars) {
     auto str = val_pair.first;
     for (auto &int_pair : val_pair.second) {
       stream << "String [" << str << "], Int pair : [" << int_pair.first << ", " << int_pair.second << "]" << std::endl;
     }
   }
   return stream.str();
}