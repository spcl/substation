#pragma once


#include "metal.hpp"

namespace MetaHelpers {

template <typename list, typename elem>
using contains = metal::distinct<metal::find<list, elem>, metal::size<list>>;

template <typename l, typename r>
using subtract = metal::remove_if<l, metal::partial<metal::lambda<contains>, r>>;

template <typename l, typename r>
using intersectPair = subtract<l, subtract<l, r>>;

template <typename... sets>
using intersect = metal::fold_left<metal::lambda<intersectPair>, sets...>;

template <typename l, typename r>
using setUnionPair = metal::join<l, subtract<r, l>>;

template <typename... sets>
using setUnion = metal::fold_left<metal::lambda<setUnionPair>, sets...>;

template <typename list>
using enumerate = metal::transpose<metal::list<metal::indices<list>, list>>;

template <typename list, typename elems>
using indicesOfElemsInList = metal::transform<metal::partial<metal::lambda<metal::find>, list>, elems>;


// isSequential
template <typename list>
using isSequentialNotEmpty = metal::same<metal::iota<metal::front<list>, metal::inc<metal::sub<metal::back<list>, metal::front<list>>>>, list>;

template <typename list> struct isSequentialOrEmpty {};
template <> struct isSequentialOrEmpty<metal::list<>> {
    using type = metal::true_;
};
template <typename... elems> struct isSequentialOrEmpty<metal::list<elems...>> {
    using type = isSequentialNotEmpty<metal::list<elems...>>; 
};

template <typename list>
using isSequential = metal::eval<isSequentialOrEmpty<list>>;

// end of isSequential

template <typename list>
using product = metal::apply<metal::lambda<metal::mul>, list>;

template <typename dim>
using dimValue = metal::number<dim::value>;

template <typename dims>
using dimsProduct = product<metal::transform<metal::lambda<dimValue>, dims>>;

// stride

template <typename dims, typename subDims>
using strideNotEmpty = dimsProduct<metal::drop<dims, metal::inc<metal::back<subDims>>>>;

template <typename dims, typename subDims> struct strideHelper {};
template <typename dims> struct strideHelper<dims, metal::list<>> {
    using type = metal::number<1>;
};
template <typename dims, typename... elems> struct strideHelper<dims, metal::list<elems...>> {
    using type = strideNotEmpty<dims, metal::list<elems...>>; 
};

template <typename dims, typename subDims>
using stride = metal::eval<strideHelper<dims, subDims>>;

// end of stride

template <typename layout, typename elem>
using elemStride = stride<layout, indicesOfElemsInList<layout, metal::list<elem>>>;

} // namespace
