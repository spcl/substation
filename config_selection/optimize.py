import argparse
import sys
import itertools
import pickle

import ops
from ops import load_ops, Variables, freeze_dict, layout_len, layout_iterator

import numpy as np
import networkx as nx


def get_linear_order(fwd_ops, bwd_ops):
    """Returns an ordered list of all operators for forward and backprop."""
    fwd_ops_order = [
        fwd_ops['QKV-fused'],
        fwd_ops['QKV-split'],
        fwd_ops['aib'],
        fwd_ops['QKT'],
        fwd_ops['softmax'],
        fwd_ops['gamma'],
        fwd_ops['out'],
        fwd_ops['bdrln1'],
        fwd_ops['lin1'],
        fwd_ops['bad'],
        fwd_ops['lin2'],
        fwd_ops['bdrln2']
    ]
    bwd_ops_order = [
        bwd_ops['bsb'],
        bwd_ops['blnrd1'],
        bwd_ops['dXlin2'],
        bwd_ops['dWlin2'],
        bwd_ops['bdrlb'],
        bwd_ops['dXlin1'],
        bwd_ops['dWlin1'],
        bwd_ops['ebsb'],
        bwd_ops['blnrd2'],
        bwd_ops['baob'],
        bwd_ops['dWout'],
        bwd_ops['dXout'],
        bwd_ops['dX1gamma'],
        bwd_ops['dX2gamma'],
        bwd_ops['bs'],
        bwd_ops['dX1QKT'],
        bwd_ops['dX2QKT'],
        bwd_ops['QKV-merge'],
        bwd_ops['dXQKV-fused'],
        bwd_ops['dWQKV-fused'],
        bwd_ops['baib'],
        bwd_ops['bei']
    ]
    return fwd_ops_order, bwd_ops_order


def build_graph(ops, vars):
    """Generate a graph of operators.

    Each node corresponds to an operator.
    There is an edge whenever one operator produces a variable that another
    consumes. The edge is labeled with all variables produced by the first
    operator and consumed by the second.

    """
    G = nx.DiGraph()
    # Add all nodes.
    for opname, op in ops.items():
        G.add_node(opname, op=op)
    # Add edges.
    for op1, op2 in itertools.product(ops.values(), ops.values()):
        vars_between = vars.vars_between(op1, op2)
        if vars_between:
            G.add_edge(op1.name, op2.name, vars=vars_between)
    return G


def concatenate_graphs(op_graphs):
    """Concatenate a list of operator graphs.

    The graphs will be concatenated in the order given. Each graph
    must have exactly one node with in-degree 0 and one node with
    out-degree 0. An edge with no variables will be added from the
    node with out-degree 0 to the node with in-degree 0 in the next
    graph.

    The nodes in the each graph must be unique.

    """
    new_G = nx.DiGraph()
    for G in op_graphs:
        new_G.add_edges_from(G.edges(data=True))
    for G, next_G in zip(op_graphs, op_graphs[1:]):
        end_node = next(node for node, out_degree in G.out_degree() if out_degree == 0)
        start_node = next(node for node, in_degree in next_G.in_degree() if in_degree == 0)
        new_G.add_edge(end_node, start_node, vars=None, concat=True)
    return new_G


def prune_graph_forward(opG):
    """Prune edges for optimizing the forward prop graph."""
    pass  # Nothing to prune.


def prune_graph_backward(opG):
    """Prune edges for optimizing the backprop graph."""
    opG.remove_edge('combined_dX1gamma_dX2gamma', 'combined_QKV-merge_baib')


def add_sssp_edges_for_op(ssspG, vars, op, index, in_vars, out_vars, binding,
                          split_idx, prev_split_idx=None):
    """Add edges for op to an SSSP graph."""
    prev_split_idx = prev_split_idx or split_idx
    # For concatenation edges, where there are no variables.
    layouts = vars.get_valid_unique_layouts(
        op.name, tuple(in_vars) + tuple(out_vars), binding=freeze_dict(binding))
    num_layouts = layout_len(layouts)
    print(f'{op.name}: Input vars: {in_vars} | output vars {out_vars} | {num_layouts} layouts')
    for cfg_idx in range(num_layouts):
        cfg_binding = dict(binding)
        for var in in_vars | out_vars:
            cfg_binding = vars.set_var_binding(cfg_binding, var, layouts[var][cfg_idx])
        cfg = vars.get_op_config_from_binding(op.name, cfg_binding)
        in_cfg = freeze_dict({var: cfg[var] for var in in_vars})
        in_node = (f'{prev_split_idx}_{index}', in_cfg)
        if in_node not in ssspG.nodes and index > 0:
            # The first op needs to add its sources, so not an error.
            raise RuntimeError(f'{op.name} trying to add edge but source {in_node} does not exist!')
        out_cfg = freeze_dict({var: cfg[var] for var in out_vars})
        out_node = (f'{split_idx}_{index+1}', out_cfg)
        weight = op.get_min_config(cfg).time
        ssspG.add_edge(in_node, out_node,
                       weight=weight,
                       cfg=cfg, op=op)
        #print(f'{op.name}: adding edge {in_node}->{out_node} weight={weight}')


def extend_sssp_graph(ssspG, vars, ops, opG, node_order, index,
                      input_vars, output_vars, binding,
                      split_vars, split_idx='0', prev_split_idx=None):
    """Add operators to the SSSP graph.

    The first time a variable in split_vars is encountered, instead of
    constructing the SSSP graph with a single set of layouts, each layout
    for that variable will be fixed and the remainder of the SSSP graph
    duplicated for each layout.

    """
    split_vars = set(split_vars)
    for i in range(len(node_order)):
        prev_split_idx = prev_split_idx or split_idx
        op = ops[node_order[i]]
        vars_to_split_on = split_vars & op.all_vars
        split_vars -= vars_to_split_on
        if op.name in input_vars:
            in_vars = set(input_vars[op.name])
        else:
            in_vars = opG.edges[(node_order[i-1], node_order[i])]['vars']
        if op.name in output_vars:
            out_vars = set(output_vars[op.name])
        else:
            out_vars = opG.edges[(node_order[i], node_order[i+1])]['vars']
        # Check if this is a concatenation edge.
        if i > 0:
            in_concat = 'concat' in opG.edges[(node_order[i-1], node_order[i])]
        else:
            in_concat = False
        if i < len(node_order) - 1:
            out_concat = 'concat' in opG.edges[(node_order[i], node_order[i+1])]
        else:
            out_concat = False
        if vars_to_split_on:
            if in_concat:
                # Add edges from concat node.
                # The output side will be handled in the recursive call.
                in_layouts = vars.get_valid_unique_layouts(
                    op.name, tuple(in_vars), binding=freeze_dict(binding))
                concat_node = f'{prev_split_idx}_{index+1}_concat'
                for layout in layout_iterator(in_layouts):
                    cfg_binding = vars.update_binding_from_cfg(binding, layout)
                    cfg = vars.get_op_config_from_binding(op.name, cfg_binding)
                    in_cfg = freeze_dict({var: cfg[var] for var in in_vars})
                    in_node = (f'{prev_split_idx}_{index+1}', in_cfg)
                    ssspG.add_edge(concat_node, in_node,
                                   weight=0.0, cfg=None, op=None)
                    #print(f'{op.name} adding in concat edge {concat_node}->{in_node}')
            layouts = vars.get_valid_unique_layouts(
                op.name, tuple(vars_to_split_on), binding=freeze_dict(binding))
            print(f'Splitting SSSP graph on variables {", ".join(vars_to_split_on)}, {layout_len(layouts)} layouts')
            # Make sure the subsequent call picks up the right variables.
            input_vars[op.name] = in_vars
            for cfg_idx, cfg in enumerate(layout_iterator(layouts)):
                split_binding = vars.update_binding_from_cfg(binding, cfg)
                extend_sssp_graph(ssspG, vars, ops, opG, node_order[i:],
                                  index + i, input_vars, output_vars,
                                  split_binding, split_vars,
                                  f'{cfg_idx}_{split_idx}', split_idx)
            # The rest of the graph was generated by the recursive calls.
            break
        else:
            if in_concat:
                # Add edges from concat node to the input layouts for op.
                in_layouts = vars.get_valid_unique_layouts(
                    op.name, tuple(in_vars), binding=freeze_dict(binding))
                concat_node = f'{prev_split_idx}_{index+i}_concat'
                for layout in layout_iterator(in_layouts):
                    cfg_binding = vars.update_binding_from_cfg(binding, layout)
                    cfg = vars.get_op_config_from_binding(op.name, cfg_binding)
                    in_cfg = freeze_dict({var: cfg[var] for var in in_vars})
                    in_node = (f'{prev_split_idx}_{index+i}', in_cfg)
                    ssspG.add_edge(concat_node, in_node,
                                   weight=0.0, cfg=None, op=None)
                    #print(f'{op.name}: adding in concat edge {concat_node}->{in_node}')
            add_sssp_edges_for_op(
                ssspG, vars, op, index + i, in_vars, out_vars,
                binding, split_idx, prev_split_idx)
            if out_concat:
                # Add edges from output layouts of op to concat node.
                out_layouts = vars.get_valid_unique_layouts(
                    op.name, tuple(out_vars), binding=freeze_dict(binding))
                concat_node = f'{split_idx}_{index+i+1}_concat'
                for layout in layout_iterator(out_layouts):
                    cfg_binding = vars.update_binding_from_cfg(binding, layout)
                    cfg = vars.get_op_config_from_binding(op.name, cfg_binding)
                    out_cfg = freeze_dict({var: cfg[var] for var in out_vars})
                    out_node = (f'{split_idx}_{index+i+1}', out_cfg)
                    ssspG.add_edge(out_node, concat_node,
                                   weight=0.0, cfg=None, op=None)
                    #print(f'{op.name}: adding out concat edge {out_node}->{concat_node}')
        # Reset since we are now past the split point.
        prev_split_idx = None


def build_sssp_graph(ops, opG, vars, start_op, input_vars, output_vars, binding,
                     split_vars=None):
    """Generate an SSSP graph for optimizing operator layout.

    Currently this only works when ops are for forward prop.

    Currently this requires pruning residual connections/etc. and more
    complex interactions.

    """
    split_vars = split_vars or []
    G = nx.DiGraph()
    # Generate the order operators will be added to the graph.
    # Note this presently breaks for backprop.
    node_order = list(nx.dfs_preorder_nodes(opG, start_op))
    # Add main edges for each operator.
    extend_sssp_graph(G, vars, ops, opG, node_order, 0,
                      input_vars, output_vars, binding,
                      split_vars)
    # Add edges from source.
    for node in [node for node, in_degree in G.in_degree() if in_degree == 0]:
        G.add_edge('source', node,
                   weight=0.0, cfg=None, op=None)
        #print(f'source: adding edge source->{node}')
    # Add edges to target.
    for node in [node for node, out_degree in G.out_degree() if out_degree == 0]:
        # This ensures we only add edges from nodes that are for the last op.
        # Not sure if we really need this check.
        if next(iter(G.in_edges(node, data=True)))[2]['op'].name == node_order[-1]:
            G.add_edge(node, 'target',
                       weight=0.0, cfg=None, op=None)
            #print(f'target: adding edge {node}->target')
    print(f'SSSP graph contains {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')
    return G


def bind_forward_vars(vars, ssspG, sssp_configs, binding):
    """Generate bindings for variables from a forward prop configuration."""
    for edge in zip(sssp_configs, sssp_configs[1:]):
        if ssspG.edges[edge]['cfg'] is None: continue
        for var, layout in ssspG.edges[edge]['cfg'].items():
            if var == 'SB2':
                # Hack for now: Do not explicitly bind SB2.
                continue
            binding = vars.set_var_binding(binding, var, layout)
    # Hack for now: Manually bind VV to match KK.
    # This should be inferred automatically, but we implicitly cut the
    # VV->gamma edge.
    binding = vars.set_var_binding(binding, 'VV', binding['KK'])

    print('Bound variables after forward optimization:')
    for var, layout in binding.items():
        if layout is not None:
            print(f'{var}: {layout}')

    return binding


def bind_backward_vars(vars, ssspG, sssp_configs, binding):
    """Generate bindings for variables from a backprop configuration."""
    for edge in zip(sssp_configs, sssp_configs[1:]):
        if ssspG.edges[edge]['cfg'] is None: continue
        for var, layout in ssspG.edges[edge]['cfg'].items():
            if var in ['DX', 'DSB2']:
                # Hack for now: Do not explicitly bind these.
                continue
            binding = vars.set_var_binding(binding, var, layout)

    print('Bound variables after backward optimization:')
    for var, layout in binding.items():
        if layout is not None:
            print(f'{var}: {layout}')

    return binding

def bind_combined_vars(vars, ssspG, sssp_configs, binding):
    """Generate bindings for variables from a configuration."""
    for edge in zip(sssp_configs, sssp_configs[1:]):
        if ssspG.edges[edge]['cfg'] is None: continue
        for var, layout in ssspG.edges[edge]['cfg'].items():
            binding = vars.set_var_binding(binding, var, layout)

    print('Bound variables after optimization:')
    for var, layout in binding.items():
        if layout is not None:
            print(f'{var}: {layout}')

    return binding


def optimize_configurations(fwd_ops, bwd_ops, graph_order,
                            split_vars=None):
    """Find the optimal end-to-end configuration.

    If fp_first is True, optimize over forward propagation first. By
    default, backpropagation is optimized first.

    """
    vars = Variables(fwd_ops, bwd_ops, ops.same_vars,
                     ops.fwd_combined_operators, ops.bwd_combined_operators)
    print('Set up Variables')
    fwd_opG = build_graph(vars.fwd_ops, vars)
    prune_graph_forward(fwd_opG)
    bwd_opG = build_graph(vars.bwd_ops, vars)
    prune_graph_backward(bwd_opG)
    binding = vars.empty_var_binding()
    if graph_order == 'fp_first':
        print('Optimizing forward pass')
        fwd_ssspG = build_sssp_graph(
            vars.fwd_ops, fwd_opG, vars, ops.fwd_operators[0].name,
            {ops.fwd_operators[0].name: [ops.fwd_input_var]},
            {ops.fwd_operators[-1].name: [ops.fwd_output_var]},
            binding)
        fwd_sssp_configs = nx.shortest_path(
            fwd_ssspG, 'source', 'target', weight='weight')
        print('Forward SSSP done')
        binding = bind_forward_vars(vars, fwd_ssspG, fwd_sssp_configs, binding)
        print('Optimizing backward pass')
        bwd_ssspG = build_sssp_graph(
            vars.bwd_ops, bwd_opG, vars, ops.bwd_operators[0].name,
            {ops.bwd_operators[0].name: [ops.bwd_input_var]},
            {ops.bwd_operators[-1].name: [ops.bwd_output_var]},
            binding)
        bwd_sssp_configs = nx.shortest_path(
            bwd_ssspG, 'source', 'target', weight='weight')
        print('Backward SSSP done')
        binding = bind_backward_vars(vars, bwd_ssspG, bwd_sssp_configs, binding)
    elif graph_order == 'bp_first':
        print('Optimizing backward pass')
        bwd_ssspG = build_sssp_graph(
            vars.bwd_ops, bwd_opG, vars, ops.bwd_operators[0].name,
            {ops.bwd_operators[0].name: [ops.bwd_input_var]},
            {ops.bwd_operators[-1].name: [ops.bwd_output_var]},
            binding)
        bwd_sssp_configs = nx.shortest_path(
            bwd_ssspG, 'source', 'target', weight='weight')
        print('Backward SSSP done')
        binding = bind_backward_vars(vars, bwd_ssspG, bwd_sssp_configs, binding)
        print('Optimizing forward pass')
        fwd_ssspG = build_sssp_graph(
            vars.fwd_ops, fwd_opG, vars, ops.fwd_operators[0].name,
            {ops.fwd_operators[0].name: [ops.fwd_input_var]},
            {ops.fwd_operators[-1].name: [ops.fwd_output_var]},
            binding)
        fwd_sssp_configs = nx.shortest_path(
            fwd_ssspG, 'source', 'target', weight='weight')
        print('Forward SSSP done')
        binding = bind_forward_vars(vars, fwd_ssspG, fwd_sssp_configs, binding)
    elif graph_order == 'combined':
        print('Building combined SSSP graph')
        combined_opG = concatenate_graphs([fwd_opG, bwd_opG])
        ssspG = build_sssp_graph(
            vars.ops, combined_opG, vars,
            ops.fwd_operators[0].name,
            {ops.fwd_operators[0].name: [ops.fwd_input_var],
             ops.bwd_operators[0].name: [ops.bwd_input_var]},
            {ops.fwd_operators[-1].name: [ops.fwd_output_var],
             ops.bwd_operators[-1].name: [ops.bwd_output_var]},
            binding, split_vars=split_vars)
        sssp_configs = nx.shortest_path(
            ssspG, 'source', 'target', weight='weight')
        print('Combined SSSP done')
        binding = bind_combined_vars(vars, ssspG, sssp_configs, binding)
    else:
        raise ValueError(f'Unknown graph order {graph_order}')

    binding = vars.minimize_binding(binding)
    vars.check_binding(binding)
    configs = vars.get_operator_configs_for_binding(binding)
    print('Final bound variables:')
    for var, layout in binding.items():
        print(f'{var}: {layout}')
    print('Final operator configurations:')
    print_configurations(configs, fwd_ops, bwd_ops)

    return configs, binding, vars


def find_best_config(fwd_ops, bwd_ops):
    """Find the best configuration, *ignoring data layout requirements*."""
    configs = {}
    for op in fwd_ops.values():
        config = op.get_min_config()
        configs[op.name] = op.get_layout_for_config(config, specials=True)
    for op in bwd_ops.values():
        config = op.get_min_config()
        configs[op.name] = op.get_layout_for_config(config, specials=True)
    return configs


def print_configurations(configs, fwd_ops, bwd_ops):
    """Print basic information on operator configurations."""
    fwd_time, bwd_time = 0.0, 0.0
    for opname in fwd_ops:
        layouts = configs[opname]
        layout_strs = " ".join([f'{var}={layout}' for var, layout in layouts.items()])
        t = fwd_ops[opname].get_min_config(layouts).time
        fwd_time += t
        delta_to_best = t - fwd_ops[opname].get_min_config().time
        print(f'{opname}: {layout_strs} | Time: {t} (from best: {delta_to_best})')
    for opname in bwd_ops:
        layouts = configs[opname]
        layout_strs = " ".join([f'{var}={layout}' for var, layout in layouts.items()])
        t = bwd_ops[opname].get_min_config(layouts).time
        bwd_time += t
        delta_to_best = t - bwd_ops[opname].get_min_config().time
        print(f'{opname}: {layout_strs} | Time: {t} (from best: {delta_to_best})')
    print(f'Forward time: {fwd_time} | Backward time: {bwd_time} | Total: {fwd_time+bwd_time}')


def save_configuration(config, binding, vars, output_file):
    layouts = {
        'input': ('X', binding['X'].upper()),
        'output': ('SB2', binding['SB2'].upper()),
        'special_dims': {},
        'algorithms': {}
    }
    for opname, op in vars.unmerged_ops.items():
        if op.specials:
            if 'Implementation' in op.specials:
                # Handle tensor contraction algorithms specially.
                output_var = op.outputs[0]
                algo = config[opname]['Implementation']
                if 'default' in algo:
                    algo = 'DEFAULT'
                else:
                    algo = 'ALGO' + algo[2:]
                layouts['algorithms'][output_var.upper()] = algo
            else:
                for special in op.specials:
                    if op.name == 'softmax':
                        # Fix naming issue.
                        entry = 'SM_' + special
                    else:
                        entry = op.name + '_' + special
                    entry = entry.upper()
                    layouts['special_dims'][entry] = config[opname][special].upper()
    layouts_params_vars = [
        'WKQV', 'BKQV', 'WO', 'BO', 'S1', 'B1', 'LINB1', 'LINW1', 'S2', 'B2',
        'LINB2', 'LINW2']
    # Temporary hack until I figure out how to merge things:
    binding['BKQV'] = 'Q' + binding['BK']
    layouts['params'] = {var: binding[var].upper() for var in layouts_params_vars}
    layouts['params'] = list(layouts['params'].items())
    layouts_forward_interm_vars = [
        'KKQQVV', 'WKKWQQWVV', 'BETA', 'ALPHA', 'ATTN_DROP_MASK', 'ATTN_DROP',
        'GAMMA', 'ATT', 'DROP1MASK', 'SB1', 'SB1_LINW1', 'DROP2', 'LIN1',
        'DROP2_LINW2', 'LIN2', 'LN2', 'LN2STD', 'LN2DIFF', 'DROP2MASK',
        'DROP3MASK', 'LN1', 'LN1STD', 'LN1DIFF']
    # Likewise another merging hack: (This uses the 'J' layout.)
    binding['KKQQVV'] = 'Q' + binding['QQ']
    layouts['forward_interm'] = {var: binding[var].upper() for var in layouts_forward_interm_vars}
    layouts['forward_interm'] = list(layouts['forward_interm'].items())
    layouts_backward_interm_vars = [
        'DLN2', 'DRESID2', 'DLIN2', 'DDROP2', 'DLIN1', 'DLIN1_LINW1', 'DLN1',
        'DRESID1', 'DATT', 'DXATT', 'DGAMMA', 'DATTN_DROP', 'DBETA', 'DKKQQVV']
    layouts['backward_interm'] = {var: binding[var].upper() for var in layouts_backward_interm_vars}
    layouts['backward_interm'] = list(layouts['backward_interm'].items())

    with open(output_file, 'wb') as f:
        pickle.dump(layouts, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find optimal configurations for transformers.')
    parser.add_argument(
        'results_dir', type=str,
        help='Directory containing benchmark results')
    parser.add_argument(
        '--min_without_layout', default=False, action='store_true',
        help='Report configuration with minimum runtime, ignoring data layout')
    parser.add_argument(
        '--graph_order', type=str, choices=['fp_first', 'bp_first', 'combined'],
        default='bp_first',
        help='Order to optimize graphs in')
    parser.add_argument(
        '--split_vars', type=str, nargs='+', default=None,
        help='Variables to split combined graph on')
    parser.add_argument(
        '--output_config', type=str, default=None,
        help='Output file to save configuration to')
    args = parser.parse_args()

    print(f'Loading results from {args.results_dir}...')
    fwd_ops, bwd_ops = load_ops(args.results_dir)
    print('Results loaded')

    if args.min_without_layout:
        configs = find_best_config(fwd_ops, bwd_ops)
        print('Best configuration *ignoring data layout changes*')
        print_configurations(configs, fwd_ops, bwd_ops)
        sys.exit(0)

    if args.split_vars:
        args.split_vars = [x.upper() for x in args.split_vars]
    config, binding, vars = optimize_configurations(
        fwd_ops, bwd_ops, args.graph_order, args.split_vars)
    if args.output_config:
        save_configuration(config, binding, vars, args.output_config)
