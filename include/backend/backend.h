#ifndef BACKEND_H
#define BACKEND_H

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif
#if defined(_WIN32)
#include <windows.h>
#endif

#include <format>
#include <iostream>

namespace backend {


} // namespace backend

#endif //BACKEND_H
