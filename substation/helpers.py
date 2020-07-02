""" Helper functions (mostly to construct SDFGs). """

import dace


def create_einsum(state: dace.SDFGState,
                  map_ranges,
                  code,
                  inputs,
                  outputs=None,
                  wcr_outputs=None
                  ):
    outputs = outputs or []
    wcr_outputs = wcr_outputs or []
    inpdict = {access_node.data: access_node for access_node, _ in inputs}
    outdict = {access_node.data: access_node for access_node, _ in
               (outputs + wcr_outputs)}

    input_memlets = {
        (access_node.data + "_inp"): dace.Memlet.simple(access_node.data,
                                                        access_range)
        for access_node, access_range in inputs}

    output_memlets = {
        (access_node.data + "_out"): dace.Memlet.simple(access_node.data,
                                                        access_range)
        for access_node, access_range in outputs}

    wcr_output_memlets = {
        (access_node.data + "_out"): dace.Memlet.simple(access_node.data,
                                                        access_range,
                                                        wcr_str='lambda x, y: x + y')
        for access_node, access_range in wcr_outputs}

    state.add_mapped_tasklet(
        name="einsum_tasklet",
        input_nodes=inpdict,
        output_nodes=outdict,
        map_ranges=map_ranges,
        inputs=input_memlets,
        code=code,
        outputs={**output_memlets, **wcr_output_memlets},
        external_edges=True
    )
