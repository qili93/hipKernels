#pragma once

#include <unordered_map>

class Head {
  public:
    static std::unordered_map<std::string, std::unordered_map<int, int>>& GetAll() {
        static std::unordered_map<std::string, std::unordered_map<int, int>> static_vars;
        return static_vars;
    }
};

class HIPCC {
  public:
   void Run();
   std::string GetString();
};

class Clang {
  public:
   void Run();
   std::string GetString();
};

class CXX {
  public:
   void Run();
   std::string GetString();
};
