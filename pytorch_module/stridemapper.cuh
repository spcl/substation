#pragma once

template <int Size1, int Size2, int Size3, int Layout1, int Layout2, int Layout3>
struct StrideMapper3 {
    static_assert(0 <= Layout1 && Layout1 < 3);
    static_assert(0 <= Layout2 && Layout2 < 3);
    static_assert(0 <= Layout3 && Layout3 < 3);
    static_assert(Layout1 != Layout2);
    static_assert(Layout2 != Layout3);
    static_assert(Layout1 != Layout3);
    
    enum { RealSize1 = Layout1 == 0 ? Size1 : (Layout2 == 0 ? Size2 : Size3) };
    enum { RealSize2 = Layout1 == 1 ? Size1 : (Layout2 == 1 ? Size2 : Size3) };
    enum { RealSize3 = Layout1 == 2 ? Size1 : (Layout2 == 2 ? Size2 : Size3) };
    
    enum { RealStride1 = RealSize2 * RealSize3 };
    enum { RealStride2 = RealSize3 };
    enum { RealStride3 = 1 };
    
    enum { Stride1 = Layout1 == 0 ? RealStride1 : (Layout1 == 1 ? RealStride2 : RealStride3) };
    enum { Stride2 = Layout2 == 0 ? RealStride1 : (Layout2 == 1 ? RealStride2 : RealStride3) };
    enum { Stride3 = Layout3 == 0 ? RealStride1 : (Layout3 == 1 ? RealStride2 : RealStride3) };
};
