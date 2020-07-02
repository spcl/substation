from dace.transformation import pattern_matching
from dace import nodes, registry, SDFGState


@registry.autoregister_params(singlestate=True, strict=True)
class MergeSourceSinkArrays(pattern_matching.Transformation):
    """ Merge duplicate arrays that are source/sink nodes. """

    _array1 = nodes.AccessNode("_")

    @staticmethod
    def expressions():
        # Matching
        #   o  o

        g = SDFGState()
        g.add_node(MergeSourceSinkArrays._array1)
        return [g]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        arr1_id = candidate[MergeSourceSinkArrays._array1]
        arr1 = graph.node(arr1_id)

        # Ensure array is either a source or sink node
        src_nodes = graph.source_nodes()
        sink_nodes = graph.sink_nodes()
        if arr1 in src_nodes:
            nodes_to_consider = src_nodes
        elif arr1 in sink_nodes:
            nodes_to_consider = sink_nodes
        else:
            return False

        # Ensure there are more nodes with the same data
        other_nodes = [graph.node_id(n) for n in nodes_to_consider
                       if isinstance(n, nodes.AccessNode) and
                       n.data == arr1.data and n != arr1]
        if len(other_nodes) == 0:
            return False

        # Ensure arr1 is the first node to avoid further duplicates
        nid = min(other_nodes)
        if nid < arr1_id:
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        arr = graph.node(candidate[MergeSourceSinkArrays._array1])
        if arr in graph.source_nodes():
            place = 'source'
        else:
            place = 'sink'
        return '%s array %s' % (place, arr.data)

    def apply(self, sdfg):
        graph = sdfg.node(self.state_id)
        array = graph.node(self.subgraph[MergeSourceSinkArrays._array1])
        if array in graph.source_nodes():
            src_node = True
            nodes_to_consider = graph.source_nodes()
            edges_to_consider = lambda n: graph.out_edges(n)
        else:
            src_node = False
            nodes_to_consider = graph.sink_nodes()
            edges_to_consider = lambda n: graph.in_edges(n)

        for node in nodes_to_consider:
            if node == array:
                continue
            if not isinstance(node, nodes.AccessNode):
                continue
            if node.data != array.data:
                continue
            for edge in list(edges_to_consider(node)):
                if src_node:
                    graph.add_edge(array, edge.src_conn, edge.dst,
                                   edge.dst_conn, edge.data)
                else:
                    graph.add_edge(edge.src, edge.src_conn, array,
                                   edge.dst_conn, edge.data)
                graph.remove_edge(edge)
            graph.remove_node(node)
