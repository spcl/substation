import argparse
import sys
import itertools
import pickle

import ops
from ops import load_ops, Variables

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


def prune_graph_forward(opG):
    """Prune edges for optimizing the forward prop graph."""
    pass  # Nothing to prune.


def prune_graph_backward(opG):
    """Prune edges for optimizing the backprop graph."""
    opG.remove_edge('combined_dX1gamma_dX2gamma', 'combined_QKV-merge_baib')


def build_sssp_graph(ops, opG, vars, start_op, input_var, output_var, binding):
    """Generate an SSSP graph for optimizing operator layout.

    Currently this only works when ops are for forward prop.

    Currently this requires pruning residual connections/etc. and more
    complex interactions.

    """
    G = nx.DiGraph()
    # Add source/target nodes for SSSP.
    G.add_node('source')
    G.add_node('target')
    # Generate the order operators will be added to the graph.
    # Note this presently breaks for backprop.
    node_order = list(nx.dfs_preorder_nodes(opG, start_op))
    # Add edges from source.
    layouts = vars.get_valid_unique_layouts(
        node_order[0], (input_var,),
        cfg=vars.get_op_config_from_binding(node_order[0], binding, hashable=True))
    for cfg in layouts[input_var]:
        out_cfg = {input_var: cfg}
        out_cfg = tuple(sorted(out_cfg.items()))
        G.add_edge('source', (0, out_cfg),
                   weight=0.0, cfg=None, op=None)
        print(f'source: adding edge source->{(0, {input_var: cfg})}')
    # Add main edges for each operator.
    for i in range(len(node_order)):
        op = ops[node_order[i]]
        if i == 0:
            in_vars = set([input_var])
        else:
            in_vars = opG.edges[(node_order[i-1], node_order[i])]['vars']
        if i == len(node_order) - 1:
            out_vars = set([output_var])
        else:
            out_vars = opG.edges[(node_order[i], node_order[i+1])]['vars']
        layouts = vars.get_valid_unique_layouts(
            op.name, tuple(in_vars) + tuple(out_vars),
            cfg=vars.get_op_config_from_binding(op.name, binding, hashable=True))
        num_layouts = len(layouts[next(iter(in_vars))])
        print(f'{op.name}: Input vars: {in_vars} | output vars: {out_vars} | {num_layouts} layouts')
        for cfg_idx in range(num_layouts):
            in_cfg = {var: layouts[var][cfg_idx] for var in in_vars}
            in_cfg = tuple(sorted(in_cfg.items()))
            in_node = (i, in_cfg)
            if in_node not in G.nodes:
                # TODO: This should be fixed in get_valid_unique_layouts.
                continue
                #raise RuntimeError(f'{op.name} trying to add edge, but source {(i, in_cfg)} does not exist!')
            out_cfg = {var: layouts[var][cfg_idx] for var in out_vars}
            out_cfg = tuple(sorted(out_cfg.items()))
            if i == len(node_order) - 1:
                # For last node, all edges go to the target.
                out_node = 'target'
            else:
                out_node = (i+1, out_cfg)
            cfg = {var: layouts[var][cfg_idx] for var in in_vars | out_vars}
            G.add_edge(in_node, out_node,
                       weight=op.get_min_config(cfg).time,
                       cfg=cfg, op=op)
            print(f'{op.name}: adding edge {in_node}->{out_node}')
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


def optimize_configurations(fwd_ops, bwd_ops):
    """Find the optimal end-to-end configuration."""
    vars = Variables(fwd_ops, bwd_ops, ops.same_vars,
                     ops.fwd_combined_operators, ops.bwd_combined_operators)
    print('Set up Variables')
    fwd_opG = build_graph(vars.fwd_ops, vars)
    prune_graph_forward(fwd_opG)
    bwd_opG = build_graph(vars.bwd_ops, vars)
    prune_graph_backward(bwd_opG)
    binding = vars.empty_var_binding()
    print('Optimizing forward pass')
    fwd_ssspG = build_sssp_graph(
        vars.fwd_ops, fwd_opG, vars, ops.fwd_operators[0].name,
        ops.fwd_input_var, ops.fwd_output_var, binding)
    fwd_sssp_configs = nx.shortest_path(
        fwd_ssspG, 'source', 'target', weight='weight')
    print('Forward SSSP done')
    binding = bind_forward_vars(vars, fwd_ssspG, fwd_sssp_configs, binding)
    print('Optimizing backward pass')
    bwd_ssspG = build_sssp_graph(
        vars.bwd_ops, bwd_opG, vars, ops.bwd_operators[0].name,
        ops.bwd_input_var, ops.bwd_output_var, binding)
    bwd_sssp_configs = nx.shortest_path(
        bwd_ssspG, 'source', 'target', weight='weight')
    print('Backward SSSP done')
    binding = bind_backward_vars(vars, bwd_ssspG, bwd_sssp_configs, binding)

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
        'layout_input': ('X', binding['X'].upper()),
        'layout_output': ('SB2', binding['SB2'].upper()),
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
    layouts['layouts_params'] = {var: binding[var].upper() for var in layouts_params_vars}
    layouts['layouts_params'] = list(layouts['layouts_params'].items())
    layouts_forward_interm_vars = [
        'KKQQVV', 'WKKWQQWVV', 'BETA', 'ALPHA', 'ATTN_DROP_MASK', 'ATTN_DROP',
        'GAMMA', 'ATT', 'DROP1MASK', 'SB1', 'SB1_LINW1', 'DROP2', 'LIN1',
        'DROP2_LINW2', 'LIN2', 'LN2', 'LN2STD', 'LN2DIFF', 'DROP2MASK',
        'DROP3MASK', 'LN1', 'LN1STD', 'LN1DIFF']
    # Likewise another merging hack: (This uses the 'J' layout.)
    binding['KKQQVV'] = 'Q' + binding['QQ']
    layouts['layout_forward_interm'] = {var: binding[var].upper() for var in layouts_forward_interm_vars}
    layouts['layout_forward_interm'] = list(layouts['layout_forward_interm'].items())
    layouts_backward_interm_vars = [
        'DLN2', 'DRESID2', 'DLIN2', 'DDROP2', 'DLIN1', 'DLIN1_LINW1', 'DLN1',
        'DRESID1', 'DATT', 'DXATT', 'DGAMMA', 'DATTN_DROP', 'DBETA', 'DKKQQVV']
    layouts['layout_backward_interm'] = {var: binding[var].upper() for var in layouts_backward_interm_vars}
    layouts['layout_backward_interm'] = list(layouts['layout_backward_interm'].items())

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

    config, binding, vars = optimize_configurations(fwd_ops, bwd_ops)
    if args.output_config:
        save_configuration(config, binding, vars, args.output_config)
