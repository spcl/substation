#pragma once

#include <iostream>
#include <cuda_fp16.h>
#include <cuda.h>

#define CHECK(expr) do {\
    auto err = (expr);\
    if (err != 0) {\
        std::cerr << "ERROR " << err << ": " << __FILE__ << ":" << __LINE__ << ": " << #expr << "\n"; \
        abort(); \
    }\
} while(0)

template <typename Real> struct AccTypeImpl { using value = Real; };
template <> struct AccTypeImpl<half> { using value = float; };

template <typename Real>
using AccType = typename AccTypeImpl<Real>::value;

template <typename T>
class ScopeGuard {
public:
    ScopeGuard(T func) : run(true), func(func) {}
    ScopeGuard(ScopeGuard<T>&& rhs)
        : run(rhs.run)
        , func(std::move(rhs.func))
    { rhs.run = false; }
    ~ScopeGuard() { if (run) func(); }
    void commit() { run = false; }
private:
    ScopeGuard(const ScopeGuard<T>& rhs) = delete;
    void operator=(const ScopeGuard<T>& rhs) = delete;
    
    bool run;
    T func;
};

template <typename T>
ScopeGuard<T> makeScopeGuard(T func) {
    return ScopeGuard<T>(func);
}

#define CAT2(x, y) x ## y
#define CAT(x, y) CAT2(x, y)
#define HOLD(x) auto CAT(unique_holder, __LINE__) = (x)



