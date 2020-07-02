import ast
import astunparse
import dace
from dace.graph.nodes import CodeNode, LibraryNode, Reduce
from dace.sdfg import Scope
from dace.symbolic import pystr_to_symbolic
from dace.libraries.blas import MatMul, Transpose
import sympy
import sys
from typing import Any, Dict, List, Union

iprint = lambda *args: print(*args)

# Do not use O(x) or Order(x) in sympy, it's not working as intended
bigo = sympy.Function('bigo')


def count_moved_data(sdfg: dace.SDFG, symbols: Dict[str, Any] = None) -> int:
    result = 0
    symbols = symbols or {}
    for state in sdfg.nodes():
        result += count_moved_data_state(state, symbols)
    return result


def count_moved_data_state(state: dace.SDFGState, symbols: Dict[str,
                                                                Any]) -> int:
    stree_root = state.scope_tree()[None]
    sdict = state.scope_dict(node_to_children=True)
    result = 0

    edges_counted = set()

    for node in sdict[None]:
        node_result = 0
        if isinstance(node, (CodeNode, LibraryNode, Reduce)):
            inputs = sum(e.data.num_accesses for e in state.in_edges(node)
                         if e not in edges_counted)
            outputs = sum(e.data.num_accesses for e in state.out_edges(node)
                          if e not in edges_counted)
            # Do not count edges twice
            edges_counted |= set(state.all_edges(node))

            iprint(
                type(node).__name__, node, 'inputs:', inputs, 'outputs:',
                outputs)
            node_result += inputs + outputs
        elif isinstance(node, dace.nodes.EntryNode):
            # Gather inputs from entry node
            inputs = sum(e.data.num_accesses for e in state.in_edges(node)
                         if e not in edges_counted)
            # Do not count edges twice
            edges_counted |= set(state.in_edges(node))
            # Gather outputs from exit node
            exit_node = state.exit_nodes(node)[0]
            outputs = sum(e.data.num_accesses
                          for e in state.out_edges(exit_node)
                          if e not in edges_counted)
            edges_counted |= set(state.out_edges(exit_node))
            iprint('Scope',
                   type(node).__name__, node, 'inputs:', inputs, 'outputs:',
                   outputs)
            node_result += inputs + outputs
        result += node_result
    return result


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('USAGE: %s <SDFG FILE>' % sys.argv[0])
        exit(1)

    sdfg = dace.SDFG.from_file(sys.argv[1])
    print('Propagating memlets')
    dace.propagate_labels_sdfg(sdfg)
    print('Counting data movement')
    dm = count_moved_data(sdfg)
    print('Total data movement', dm)
