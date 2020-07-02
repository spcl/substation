#pragma once

#include "metahelpers.h"

namespace EinsumParser {
    
using namespace MetaHelpers;

template <typename A, typename B, typename C>
struct EinsumParser {
    
    using batchVars = intersect<A, B, C>;
    using sumVars = subtract<intersect<A, B>, C>;
    using aOnlyVars = subtract<subtract<A, sumVars>, batchVars>;
    using bOnlyVars = subtract<subtract<B, sumVars>, batchVars>;
       
    using aBatch = indicesOfElemsInList<A, batchVars>;
    using aSum = indicesOfElemsInList<A, sumVars>;
    using aOnly = indicesOfElemsInList<A, aOnlyVars>;
    
    using bBatch = indicesOfElemsInList<B, batchVars>;
    using bSum = indicesOfElemsInList<B, sumVars>;
    using bOnly = indicesOfElemsInList<B, bOnlyVars>;
    
    using caOnly = indicesOfElemsInList<C, aOnlyVars>;
    using cbOnly = indicesOfElemsInList<C, bOnlyVars>;
    using cBatch = indicesOfElemsInList<C, batchVars>;
    
    static_assert(isSequential<aBatch>::value, "A BATCH is not sequential");
    static_assert(isSequential<aSum>::value, "A SUM is not sequential");
    static_assert(isSequential<aOnly>::value, "A ONLY is not sequential");
    static_assert(isSequential<bBatch>::value, "B SUM is not sequential");
    static_assert(isSequential<bSum>::value, "B SUM is not sequential");
    static_assert(isSequential<bOnly>::value, "B ONLY is not sequential");
    static_assert(isSequential<caOnly>::value, "C A ONLY is not sequential");
    static_assert(isSequential<cbOnly>::value, "C B ONLY is not sequential");
    static_assert(isSequential<cBatch>::value, "C BATCH is not sequential");
    
    using batch = dimsProduct<batchVars>;
    using m = dimsProduct<aOnlyVars>;
    using k = dimsProduct<sumVars>;
    using n = dimsProduct<bOnlyVars>;
    
    using sAM = stride<A, aOnly>;
    using sAK = stride<A, aSum>;
    using sAB = stride<A, aBatch>;
    
    using sBK = stride<B, bSum>;
    using sBN = stride<B, bOnly>;
    using sBB = stride<B, bBatch>;
    
    using sCM = stride<C, caOnly>;
    using sCN = stride<C, cbOnly>;
    using sCB = stride<C, cBatch>;
};

} // namespace
