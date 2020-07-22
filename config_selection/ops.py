import os.path
import functools
import itertools
import time
from collections import namedtuple, defaultdict
from load_data import *

# Benchmarked operators.
tc_fwd_ops = [
    'QKV-fused',
    'QKT',
    'gamma',
    'out',
    'lin1',
    'lin2'
]
tc_bwd_ops = [
    'dWlin2',
    'dXlin2',
    'dWlin1',
    'dXlin1',
    'dWout',
    'dXout',
    'dX2gamma',
    'dX1gamma',
    'dX2QKT',
    'dX1QKT',
    'dWQKV-fused',
    'dXQKV-fused'
]

kernel_fwd_ops = [
    'aib',
    'softmax',
    'bdrln',
    'bad'
]
kernel_bwd_ops = [
    'bsb',
    'blnrd',
    'bdrlb',
    'ebsb',
    'baob',
    'bs',
    'baib',
    'bei'
]

# Actual operators used in encoder layer.
# Fields:
# - Operator name.
# - Type of operator (kernel or tensor contraction).
# - Corresponding benchmarked operator.
# - Input variables, as (name, benchmark field name).
# - Output variables (same format).
# - Other special fields (e.g., vectorization dimension) (same format).
# Split/merge operators abuse these fields somewhat.
OperatorDef = namedtuple(
    'OperatorDef',
    ['name', 'type', 'bm', 'input', 'output', 'special'])
fwd_operators = [
    OperatorDef(
        'QKV-fused', 'tc', 'QKV-fused',
        [('WKQV', 'A'), ('X', 'B')],
        [('WKKWQQWVV', 'C')],
        ['Implementation']),
    OperatorDef(
        'QKV-split', 'split', 'QKV-fused',
        ['WKKWQQWVV'],
        ['WQQ', 'WKK', 'WVV'],
        None),
    OperatorDef(
        'aib', 'kernel', 'aib',
        [('WKK', 'wkk'), ('BK', 'bk'),
         ('WVV', 'wvv'), ('BV', 'bv'),
         ('WQQ', 'wqq'), ('BQ', 'bq')],
        [('KK', 'kk'), ('VV', 'vv'), ('QQ', 'qq')],
        ['dv', 'dt']),
    OperatorDef(
        'QKT', 'tc', 'QKT',
        [('KK', 'A'), ('QQ', 'B')],
        [('BETA', 'C')],
        ['Implementation']),
    OperatorDef(
        'softmax', 'kernel', 'softmax',
        [('BETA', 'in')],
        [('ALPHA', 'out'), ('ATTN_DROP_MASK', 'drop_mask'),
         ('ATTN_DROP', 'drop')],
        [('dv', 'vec_dim')]),
    OperatorDef(
        'gamma', 'tc', 'gamma',
        [('VV', 'A'), ('ATTN_DROP', 'B')],
        [('GAMMA', 'C')],
        ['Implementation']),
    OperatorDef(
        'out', 'tc', 'out',
        [('WO', 'A'), ('GAMMA', 'B')],
        [('ATT', 'C')],
        ['Implementation']),
    OperatorDef(
        'bdrln1', 'kernel', 'bdrln',
        [('ATT', 'drop2_linw2'), ('X', 'sb1'),
         ('S1', 's2'), ('B1', 'b2'),
         ('BO', 'linb2'), ('ATT', 'lin2')],
        [('SB1', 'sb2'), ('LN1', 'ln2'),
         ('LN1DIFF', 'ln2diff'), ('LN1STD', 'ln2std'),
         ('DROP1MASK', 'drop3mask')],
        [('dv', 'vec_dim')]),
    OperatorDef(
        'lin1', 'tc', 'lin1',
        [('SB1', 'A'), ('LINW1', 'B')],
        [('SB1_LINW1', 'C')],
        ['Implementation']),
    OperatorDef(
        'bad', 'kernel', 'bad',
        [('SB1_LINW1', 'sb1_linw1'), ('LINB1', 'linb1')],
        [('DROP2', 'drop2'), ('DROP2MASK', 'drop2mask'),
         ('LIN1', 'lin1')],
        [('dv', 'vec_dim'), ('dt', 'thread_dim')]),
    OperatorDef(
        'lin2', 'tc', 'lin2',
        [('DROP2', 'A'), ('LINW2', 'B')],
        [('DROP2_LINW2', 'C')],
        ['Implementation']),
    OperatorDef(
        'bdrln2', 'kernel', 'bdrln',
        [('DROP2_LINW2', 'drop2_linw2'), ('SB1', 'sb1'),
         ('S2', 's2'), ('B2', 'b2'),
         ('LINB2', 'linb2')],
        [('SB2', 'sb2'), ('LN2', 'ln2'),
         ('LIN2', 'lin2'), ('LN2DIFF', 'ln2diff'),
         ('LN2STD', 'ln2std'), ('DROP3MASK', 'drop3mask')],
        [('dv', 'vec_dim')])
]
fwd_input_var = 'X'  # Overall input variable.
fwd_output_var = 'SB2'  # Overall output variable.
bwd_operators = [
    OperatorDef(
        'bsb', 'kernel', 'bsb',
        [('LN2', 'ln2'), ('S2', 's2'),
         ('DSB2', 'dsb2')],
        [('DLN2', 'dln2'), ('DS2', 'ds2'),
         ('DB2', 'db2')],
        [('dv', 'vec_dim'), ('dw', 'warp_dim')]),
    OperatorDef(
        'blnrd1', 'kernel', 'blnrd',
        [('DLN2', 'dln2'), ('LN2STD', 'ln2std'),
         ('LN2DIFF', 'ln2diff'), ('DROP3MASK', 'drop3mask')],
        [('DRESID2', 'dresid2'), ('DLIN2', 'dlin2')],
        [('dv', 'vec_dim')]),
    OperatorDef(
        'dXlin2', 'tc', 'dXlin2',
        [('DLIN2', 'A'), ('LINW2', 'B')],
        [('DDROP2', 'C')],
        ['Implementation']),
    OperatorDef(
        'dWlin2', 'tc', 'dWlin2',
        [('DLIN2', 'A'), ('DROP2', 'B')],
        [('DLINW2', 'C')],
        ['Implementation']),
    OperatorDef(
        'bdrlb', 'kernel', 'bdrlb',
        [('DDROP2', 'ddrop2'), ('DROP2MASK', 'drop2mask'),
         ('LIN1', 'lin1')],
        [('DLIN1', 'dlin1'), ('DLINB1', 'dlinb1')],
        [('dv', 'vec_dim'), ('dw', 'warp_dim')]),
    OperatorDef(
        'dXlin1', 'tc', 'dXlin1',
        [('DLIN1', 'A'), ('LINW1', 'B')],
        [('DLIN1_LINW1', 'C')],
        ['Implementation']),
    OperatorDef(
        'dWlin1', 'tc', 'dWlin1',
        [('DLIN1', 'A'), ('SB1', 'B')],
        [('DLINW1', 'C')],
        ['Implementation']),
    OperatorDef(
        'ebsb', 'kernel', 'ebsb',
        [('LN1', 'ln1'), ('S1', 's1'),
         ('DRESID2', 'dresid2'), ('DLIN1_LINW1', 'dlin1_linw1'),
         ('DLIN2', 'dlin2')],
        [('DLINB2', 'dlinb2'), ('DS1', 'ds1'),
         ('DB1', 'db1'), ('DLN1', 'dln1')],
        [('dv', 'vec_dim'), ('dw', 'warp_dim')]),
    OperatorDef(
        'blnrd2', 'kernel', 'blnrd',
        [('DLN1', 'dln2'), ('LN1STD', 'ln2std'),
         ('LN1DIFF', 'ln2diff'), ('DROP1MASK', 'drop3mask')],
        [('DRESID1', 'dresid2'), ('DATT', 'dlin2')],
        [('dv', 'vec_dim')]),
    OperatorDef(
        'baob', 'kernel', 'baob',
        [('DATT', 'datt')],
        [('DBO', 'dbo')],
        ['dv', 'dw']),
    OperatorDef(
        'dXout', 'tc', 'dXout',
        [('WO', 'A'), ('DATT', 'B')],
        [('DGAMMA', 'C')],
        ['Implementation']),
    OperatorDef(
        'dWout', 'tc', 'dWout',
        [('GAMMA', 'A'), ('DATT', 'B')],
        [('DWO', 'C')],
        ['Implementation']),
    OperatorDef(
        'dX1gamma', 'tc', 'dX1gamma',
        [('VV', 'A'), ('DGAMMA', 'B')],
        [('DATTN_DROP', 'C')],
        ['Implementation']),
    OperatorDef(
        'dX2gamma', 'tc', 'dX2gamma',
        [('DGAMMA', 'A'), ('ATTN_DROP', 'B')],
        [('DVV', 'C')],
        ['Implementation']),
    OperatorDef(
        'bs', 'kernel', 'bs',
        [('ALPHA', 'alpha'), ('ATTN_DROP_MASK', 'drop_mask'),
         ('DATTN_DROP', 'drop')],
        [('DBETA', 'dbeta')],
        [('dv', 'vec_dim')]),
    OperatorDef(
        'dX1QKT', 'tc', 'dX1QKT',
        [('KK', 'A'), ('DBETA', 'B')],
        [('DQQ', 'C')],
        ['Implementation']),
    OperatorDef(
        'dX2QKT', 'tc', 'dX2QKT',
        [('QQ', 'A'), ('DBETA', 'B')],
        [('DKK', 'C')],
        ['Implementation']),
    OperatorDef(
        'QKV-merge', 'merge', ['dX2gamma', 'dX1QKT', 'dX2QKT'],
        ['DQQ', 'DKK', 'DVV'],
        ['DKKQQVV'],
        None),
    OperatorDef(
        'dXQKV-fused', 'tc', 'dXQKV-fused',
        [('WKQV', 'A'), ('DKKQQVV', 'B')],
        [('DXATT', 'C')],
        ['Implementation']),
    OperatorDef(
        'dWQKV-fused', 'tc', 'dWQKV-fused',
        [('X', 'A'), ('DKKQQVV', 'B')],
        [('DWKQV', 'C')],
        ['Implementation']),
    OperatorDef(
        'baib', 'kernel', 'baib',
        [('DKK', 'dkk'), ('DVV', 'dvv'), ('DQQ', 'dqq')],
        [('DBK', 'dbk'), ('DBV', 'dbv'), ('DBQ', 'dbq')],
        ['dv', 'dw']),
    OperatorDef(
        'bei', 'kernel', 'bei',
        [('DXATT', 'dxatt'), ('DRESID1', 'dresid1')],
        [('DX', 'dx')],
        ['dt', 'dv'])
]
bwd_input_var = 'DSB2'
bwd_output_var = 'DX'

# Some variables must use the same layout (e.g., weights and their gradients)
# for correctness. This is a list of lists of variables that must share layouts.
# This is mainly relevant when the variables are not constrained by the
# operator itself (which should be the case for e.g. dropout and masks).
same_vars = [
    (fwd_input_var, fwd_output_var),  # Ensures we can chain layers together.
    (bwd_input_var, bwd_output_var),
    # PyTorch requires this:
    (fwd_input_var, bwd_output_var),
    (fwd_output_var, bwd_input_var),
    # Weight gradients:
    ('WKQV', 'DWKQV'),
    ('BK', 'DBK'),
    ('BV', 'DBV'),
    ('BQ', 'DBQ'),
    ('WO', 'DWO'),
    ('BO', 'DBO'),
    ('S1', 'DS1'),
    ('B1', 'DB1'),
    ('LINW1', 'DLINW1'),
    ('LINB1', 'DLINB1'),
    ('LINW2', 'DLINW2'),
    ('LINB2', 'DLINB2'),
    ('S2', 'DS2'),
    ('B2', 'DB2'),
    # These need to match because they are stacked:
    # Removed because of issues with j vs k.
    #('DQQ', 'DKK', 'DVV')
]
# This augments same_vars to also require sameness between forward prop
# activations and the corresponding backward error signals.
same_vars_backprop = same_vars + [
    (fwd_input_var, bwd_output_var),
    (fwd_output_var, bwd_input_var),
    ('WKKWQQWVV', 'DKKQQVV'),
    ('KK', 'DKK'),
    ('VV', 'DVV'),
    ('QQ', 'DQQ'),
    ('BETA', 'DBETA'),
    ('ATTN_DROP', 'DATTN_DROP'),
    ('GAMMA', 'DGAMMA'),
    ('ATT', 'DATT'),
    ('DROP2', 'DDROP2'),
    ('LIN1', 'DLIN1'),
    ('LN2', 'DLN2'),
    ('SB1', 'DRESID2'),
    ('X', 'DRESID1'),
    ('LIN2', 'DLIN2'),
    ('SB1', 'DLIN1_LINW1'),
    ('LN1', 'DLN1'),
    ('X', 'DXATT'),
]
# Reprocess into sets for simplicity.
same_vars = [set(x) for x in same_vars]
same_vars_backprop = [set(x) for x in same_vars_backprop]
# Combine operators to simplify optimization.
fwd_combined_operators = None
bwd_combined_operators = [
    ('dXlin2', 'dWlin2'),
    ('dXlin1', 'dWlin1'),
    ('dXout', 'dWout', 'baob'),
    ('dX1gamma', 'dX2gamma'),
    ('dX1QKT', 'dX2QKT'),
    ('QKV-merge', 'baib'),
    ('dXQKV-fused', 'dWQKV-fused')
]


def layout_len(layouts):
    """Return the number of layouts in a layout dict."""
    return len(next(iter(layouts.values())))


def _layout_iterator_no_cfg(layouts, restrict=None):
    """Faster version of layout_iterator when cfg is None."""
    vars = list(restrict) if restrict else list(layouts.keys())
    restricted_layouts = {var: layouts[var] for var in vars}
    num_layouts = len(restricted_layouts[vars[0]])
    for i in range(num_layouts):
        yield {var: restricted_layouts[var][i] for var in vars}


def bare_layout_iterator(layouts, order=None):
    """Yield layouts without the dict.

    The order is that given by layouts.keys() if order is None.

    """
    if order is None:
        order = list(layouts.keys())
    layouts = [layouts[var] for var in order]
    yield from zip(*layouts)


def layout_iterator(layouts, restrict=None, cfg=None):
    """A generator that yields each overall layout in layouts.

    For each layout in layouts, returns a dict where the keys are variables
    and the values are the corresponding layouts.

    If restrict is not None, it should be a list of variables, and only those
    will be returned.

    If cfg is not None, it should be a dict mapping variables to layouts,
    and only configurations where variables have those layouts will be
    returned.

    """
    if cfg is None:
        yield from _layout_iterator_no_cfg(layouts, restrict=restrict)
    else:
        vars = list(restrict) if restrict else list(layouts.keys())
        num_layouts = len(layouts[vars[0]])
        for i in range(num_layouts):
            match = True
            for var, layout in cfg.items():
                if layouts[var][i] != layout:
                    match = False
                    break
                if not match: continue
                yield {var: layouts[var][i] for var in vars}


def filter_layouts(x, vars):
    """Filter an iterable to contain only vars in vars.

    If x is None, this just returns None.

    Works with lists/tuples (will filter directly) and dicts (filter keys).

    If the filtering makes x empty, returns None.

    """
    if x is None:
        return None
    vars = set(vars)
    if type(x) in (list, tuple, set):
        x = type(x)([var for var in x if var in vars])
    elif type(x) is dict:
        x = {var: x[var] for var in x if var in vars}
    else:
        raise ValueError(f'Do not know how to filter {x}')
    if not x:
        return None
    else:
        return x


def restrict_layout_configs(layouts, cfg):
    """Restrict layouts to contain only layouts that match cfg.

    cfg is a dict mapping variables to a fixed layout. It is allowable
    for cfg to have variables that are not in layouts: they will be
    ignored.

    """
    if not cfg:
        return layouts
    cfg = filter_layouts(cfg, layouts.keys())
    if not cfg:
        return layouts
    vars = list(layouts.keys())
    restrict_vars = list(cfg.keys())
    restrict_cfg = tuple(cfg[var] for var in restrict_vars)
    restrict_indices = [vars.index(var) for var in restrict_vars]
    restrict_indices_cfg = tuple(zip(restrict_indices, restrict_cfg))
    new_layouts = {var: [] for var in vars}
    for bare_layout in bare_layout_iterator(layouts, order=vars):
        if all(bare_layout[idx] == layout for idx, layout in restrict_indices_cfg):
            for var, layout in zip(vars, bare_layout):
                new_layouts[var].append(layout)
    return new_layouts


def remap_vars_by_sameness(x, y, sameness):
    """Rename vars in x to be those in y based on sameness.

    Every variable in x is mapped to the corresponding variable in y.
    It is allowable that x have fewer variables than y, but there must
    be a corresponding variable in y for every variable in x.

    """
    name_map = defaultdict(list)
    for var in x:
        for same_var in sameness[var]:
            if same_var in y:
                name_map[var].append(same_var)
        if var not in name_map:
            raise ValueError(f'Do not know how to remap {var} in {x} using {y}')
    if type(x) in (list, tuple, set):
        new_x = []
        for var in x:
            new_x += name_map[var]
        new_x = type(x)(new_x)
    elif type(x) is dict:
        new_x = {}
        for var in x:
            for new_var in name_map[var]:
                new_x[new_var] = x[var]
    else:
        raise ValueError(f'Do not know how to remap {x}')
    return new_x


def remap_subset_by_sameness(x, y, sameness):
    """Return and rename the subset of x corresponding to y.

    This is like remap_vars_by_sameness, but will return only the subset
    of x that matches y.

    """
    # Identify the common subset.
    subset = []
    for var in x:
        for same_var in sameness[var]:
            if same_var in y:
                subset.append(var)
                break
    subset_x = filter_layouts(x, subset)
    return remap_vars_by_sameness(subset_x, y, sameness)


def freeze_dict(d):
    """Return a frozen version of dict that is hashable."""
    return tuple(sorted(d.items()))


def merge_two_layouts_no_shared(layouts1, layouts2):
    """Merge two sets of layouts, having no variables in common.

    layouts1 and layouts2 should be dicts of lists of layouts for
    each variable, as returned by Operator.get_unique_layouts.

    """
    # Ensure no common variables:
    assert not (set(layouts1.keys()) & set(layouts2.keys()))
    # Essentially, for each layout in layouts1, this will add every
    # layout in layouts2 to the output.
    new_layouts = {var: [] for var in
                   itertools.chain(layouts1.keys(), layouts2.keys())}
    num_layout2s = len(next(iter(layouts2.values())))
    for layout1 in layout_iterator(layouts1):
        # Entries from layout1 are duplicated, add them all here.
        for var, layout in layout1.items():
            new_layouts[var].extend([layout]*num_layout2s)
        for layout2 in layout_iterator(layouts2):
            for var, layout in layout2.items():
                new_layouts[var].append(layout)
    return new_layouts


def get_common_vars(layouts1, layouts2, sameness):
    """Return the set of common variables between layout sets."""
    if sameness is None:
        return None
    common_vars = set()
    for var in layouts1.keys():
        for same_var in sameness[var]:
            if same_var in layouts2:
                common_vars.add(var)
                break
    return common_vars


def merge_two_layouts(layouts1, layouts2, sameness=None):
    """Merge two sets of layouts, which may have variables in common.

    If there are incompatible layouts of common variables, they are dropped.

    If sameness is not None, it is a map from variable names to all variables
    which a variable must have the same layouts as.

    """
    # Provide a default sameness map.
    if sameness is None:
        sameness = {var: [var] for var
                    in itertools.chain(layouts1.keys(), layouts2.keys())}
    # Identify common variables between the two sets, accounting for sameness.
    # For variables that need to be the same, this stores the variable from
    # layouts1.
    common_vars = get_common_vars(layouts1, layouts2, sameness)
    if not common_vars:
        return merge_two_layouts_no_shared(layouts1, layouts2)
    # First find the unique set of layouts for the common variables among the
    # two sets of layouts.
    # From this point, we drop dicts for layouts for efficiency. The order of
    # variables is given implicitly by common_vars.
    shared_layouts1 = set(zip(*[layouts1[var] for var in common_vars]))
    # Pick the subset of layouts2 that corresponds to common_vars and remap the variables.
    remapped_layouts2_subset = remap_subset_by_sameness(
        layouts2, common_vars, sameness)
    shared_layouts2 = set(zip(
        *[remapped_layouts2_subset[var] for var in common_vars]))
    shared_layouts = shared_layouts1 & shared_layouts2
    if not shared_layouts:
        raise RuntimeError(f'No common layouts to expand, {layouts1.keys()} and {layouts2.keys()}')
    # Now construct a map from the shared layouts to *every* layout in layouts2
    # that matches the shared layout.
    # This precomputation lets us avoid repeated iteration over layouts2 below.
    # This does not remap vars in the final layout, but does filter out the
    # variables that are exactly the same between layouts1 and layouts2 (without
    # considering sameness).
    new_layouts = {var: [] for var in
                   itertools.chain(layouts1.keys(), layouts2.keys())}
    layout1_vars = list(layouts1.keys())
    layout1_common_indices = [layout1_vars.index(var) for var in common_vars]
    layouts2_only_vars = set(layouts2.keys()) - common_vars
    if layouts2_only_vars:
        layout_map = {layout: [] for layout in shared_layouts}
        for shared_layout, bare_layout in zip(
                bare_layout_iterator(remapped_layouts2_subset, order=common_vars),
                bare_layout_iterator(layouts2, order=layouts2_only_vars)):
            if shared_layout in layout_map:
                layout_map[shared_layout].append(bare_layout)
        # Finally, for every layout in layouts1, add every layout in layouts2 where
        # the shared variables have matching layouts.
        for bare_layout1 in bare_layout_iterator(layouts1):
            shared_layout = tuple(bare_layout1[idx] for idx in layout1_common_indices)
            if shared_layout not in layout_map:
                # Layout does not have a match in the other set, skip.
                continue
            # The entries from layout1 are duplicated, so add them all here.
            num_layout2s = len(layout_map[shared_layout])
            for var, layout in zip(layout1_vars, bare_layout1):
                new_layouts[var].extend([layout]*num_layout2s)
            for bare_layout2 in layout_map[shared_layout]:
                for var, layout in zip(layouts2_only_vars, bare_layout2):
                    new_layouts[var].append(layout)
    else:
        # In this case, every variable in layouts2 is also in layouts1, even
        # without accounting for sameness, so we don't need a mapping.
        for bare_layout1 in bare_layout_iterator(layouts1):
            shared_layout = tuple(bare_layout1[idx] for idx in layout1_common_indices)
            if shared_layout not in shared_layouts:
                continue
            for var, layout in zip(layout1_vars, bare_layout1):
                new_layouts[var].append(layout)
    return new_layouts


def merge_layouts(layouts, sameness=None):
    """Iteratively merge layouts."""
    new_layouts = layouts[0]
    ordered_layouts = layouts[1:]
    # Order based on sets of layouts that have variables in common with what
    # we first picked, to reduce overall time.
    # Could pick a better overall order if needed (by picking new_layouts better).
    if sameness:
        ordered_layouts.sort(key=lambda x: len(get_common_vars(
            new_layouts, x, sameness)), reverse=True)
    for op_layout in ordered_layouts:
        new_layouts = merge_two_layouts(new_layouts, op_layout,
                                        sameness=sameness)
    return new_layouts


def get_compatible_layouts_pair(layouts1, layouts2, sameness):
    """Return the layouts from each set that are compatible.

    This is like merge_two_layouts, except instead of constructing the
    cross-product of valid layouts to explicitly materialize the merge,
    it returns the sets separately.

    """
    # Identify common variables, accounting for sameness.
    # This stores variables under their name in layouts1.
    common_vars = get_common_vars(layouts1, layouts2, sameness)
    if not common_vars:
        # No common variables, so nothing to filter out.
        return layouts1, layouts2
    # Find the unique sets of layouts for the common variables.
    common_layouts1 = set(zip(*[layouts1[var] for var in common_vars]))
    remapped_layouts2_subset = remap_subset_by_sameness(
        layouts2, common_vars, sameness)
    common_layouts2 = set(zip(
        *[remapped_layouts2_subset[var] for var in common_vars]))
    # Identify the layouts that match between the two.
    shared_layouts = common_layouts1 & common_layouts2
    if not shared_layouts:
        # No common layouts, these are incompatible.
        raise RuntimeError(f'No common layouts to expand, {layouts1.keys()} and {layouts2.keys()}')
    # Now filter both layout sets to contain only the shared layouts.
    # We use two stages, the first filters and then the second constructs
    # the new layout dicts. This lets us use list comprehensions for both,
    # which is way faster. (I haven't figured out how to do it in one pass
    # with list comprehensions.)
    layout1_vars = list(layouts1.keys())
    new_layouts1_tmp = [bare_layout1 for shared_layout, bare_layout1 in zip(
        bare_layout_iterator(layouts1, order=common_vars),
        bare_layout_iterator(layouts1, order=layout1_vars))
                        if shared_layout in shared_layouts]
    new_layouts1 = {}
    for i, var in enumerate(layout1_vars):
        new_layouts1[var] = [bare_layout1[i] for bare_layout1 in new_layouts1_tmp]
    layout2_vars = list(layouts2.keys())
    new_layouts2_tmp = [bare_layout2 for shared_layout, bare_layout2 in zip(
        bare_layout_iterator(remapped_layouts2_subset, order=common_vars),
        bare_layout_iterator(layouts2, order=layout2_vars))
                        if shared_layout in shared_layouts]
    new_layouts2 = {}
    for i, var in enumerate(layout2_vars):
        new_layouts2[var] = [bare_layout2[i] for bare_layout2 in new_layouts2_tmp]

    return new_layouts1, new_layouts2


def get_compatible_layouts(layouts, sameness):
    """Return the set of layouts that is compatible among all layouts.

    This is like merge_layouts, but does not explicitly build the
    cross-product.

    """
    # We need to iterate and compare all pairs of layouts to make sure
    # information propagates. This is potentially quite slow with many
    # operators, but does avoid materializing large lists.
    new_layouts = layouts[:]
    while True:
        changed = False
        for idx1, idx2 in itertools.product(range(len(layouts)), repeat=2):
            if idx1 == idx2:
                continue  # A layout is trivially compatible with itself.
            # Normally things are symmetric, but they can not be when
            # get_common_vars differs, typically when one layout has multiple
            # variables that are the same.
            if idx2 < idx1 and (
                    get_common_vars(new_layouts[idx1], new_layouts[idx2], sameness)
                    == get_common_vars(new_layouts[idx2], new_layouts[idx1], sameness)):
                continue
            layouts1, layouts2 = new_layouts[idx1], new_layouts[idx2]
            new_layouts1, new_layouts2 = get_compatible_layouts_pair(
                layouts1, layouts2, sameness)
            # Since this only filters, we can check for changes by checking
            # whether the length changed.
            if layout_len(layouts1) != layout_len(new_layouts1):
                changed = True
                new_layouts[idx1] = new_layouts1
            if layout_len(layouts2) != layout_len(new_layouts2):
                changed = True
                new_layouts[idx2] = new_layouts2
        if not changed:
            break
    return new_layouts


class Operator:
    """Represents all data and related operators for a particular operator."""

    def __init__(self, opdef, result_dir, filter_fused=True, rename_aib=True):
        """Set up the operator.

        opdef is an OperatorDef defining the operator.
        result_dir is the director containing benchmark results.
        filter_fused controls whether certain not-feasible configurations
        are removed from fused tensor contraction operators.
        rename_aib renames configurations in aib/baib to avoid some benchmark
        simplifications.

        """
        self.name = opdef.name
        if opdef.type == 'tc':
            self.data = load_tc(os.path.join(
                result_dir, f'{opdef.bm}-result.csv'))
            if filter_fused and 'fused' in self.name:
                self._filter_fused_configs()
        elif opdef.type == 'kernel':
            self.data = load_kernel(os.path.join(
                result_dir, f'{opdef.bm}-combined.csv'))
            if rename_aib and self.name in ['aib', 'baib']:
                self._rename_aib_configs()
        else:
            raise ValueError(f'Unknown operator type {opdef.type} for {self.name}')
        if self.data is not None:
            print(f'{self.name} loaded {len(self.data)} configs')

        # For caching partial restricts up to two levels.
        self.restrict_cache = {}

        # Set up inputs/output details.
        # inputs/output are a list of input/output variables.
        self.inputs, self.outputs = [], []
        # This will rename dataframe columns appropriately.
        # When the same variable is mapped to multiple columns
        # (occurs in bdrln1), we ensure we keep only rows where the columns
        # are equal, and then drop the extra columns.
        var_map = defaultdict(list)
        for var, data_var in opdef.input:
            if data_var not in self.data.columns:
                raise ValueError(f'Input var {var}->{data_var} not in data for {self.name}')
            self.inputs.append(var)
            var_map[var].append(data_var)
        for var, data_var in opdef.output:
            if data_var not in self.data.columns:
                raise ValueError(f'Output var {var}->{data_var} not in data for {self.name}')
            self.outputs.append(var)
            var_map[var].append(data_var)
        # Build the rename map, identify columns mapped to the same variable,
        # and which columns to drop.
        rename_map = {}
        cols_to_drop = []
        for var, data_vars in var_map.items():
            # Arbitrarily map the first column to this variable.
            rename_map[data_vars[0]] = var
            if len(data_vars) > 1:
                # Restrict to rows where columns match.
                # May be a better way to do this.
                keep = self.data[data_vars[0]] == self.data[data_vars[1]]
                for data_var in data_vars[2:]:
                    keep = keep == self.data[data_var]
                self.data = self.data[keep]
                # Drop all remaining columns.
                cols_to_drop += data_vars[1:]
        # Drop and rename.
        if cols_to_drop:
            self.data.drop(columns=cols_to_drop, inplace=True)
        self.data.rename(columns=rename_map, inplace=True)

        # Convenience for accessing all variables.
        self.all_vars = set(self.inputs + self.outputs)

        # Remap specials.
        self.specials = []
        if opdef.special:
            rename_map = {}
            for special in opdef.special:
                # For simplicity, since most do not change.
                if type(special) is tuple:
                    if special[1] not in self.data.columns:
                        raise ValueError(f'Special {special[0]}->{special[1]} not in data for {self.name}')
                    rename_map[special[1]] = special[0]
                    self.specials.append(special[0])
                else:
                    if special not in self.data.columns:
                        raise ValueError(f'Special {special}->{special} not in data for {self.name}')
                    self.specials.append(special)
            if rename_map:
                self.data.rename(columns=rename_map, inplace=True)


    def get_unique_layouts(self, vars=None, cfg=None):
        """Return unique layouts for operator.

        This will return a dict, where each key is a variable and the value is
        a list of layouts. Every variable will have a list that is the same
        length, and entries at the same position correspond to the same overall
        layout.

        Note, each list may have duplicates: It is the overall layout that is
        unique.

        If vars is not None, it should be a list of variables to get unique
        layouts among. By default, it will be all input/outputs.

        If cfg is not None, it will be used to restrict the layouts to include
        only those with variables with a fixed layout (like get_min_config).
        Note these variables do *not* need to also be in vars.

        """
        if vars is None:
            vars = self.inputs + self.outputs

        if cfg:
            restrict = True
            for var, layout in cfg.items():
                restrict = restrict & (self.data[var] == layout)
            df = self.data[restrict]
        else:
            df = self.data
        df = df.drop_duplicates(vars)
        cfgs = {}
        for var in vars:
            cfgs[var] = list(df[var])
        return cfgs


    def get_min_config(self, cfg=None, cache=False):
        """Return the configuration with minimum time.

        cfg can be used to optionally restrict the minimum to be only over
        vars with a particular layout. It should be a dict mapping vars to
        the bound layout.

        If cache is True, partial restricts will be cached to speed up
        repeated calls with similar cfgs.

        """
        if cfg is None:
            return self._get_row(self.data['time'].idxmin())

        if cache:
            var_order = sorted([var for var in cfg.keys() if var not in self.specials])
            cache_vars = 2 if len(var_order) > 3 else 1
            cache_key = tuple((var, cfg[var]) for var in var_order[:cache_vars])
            if cache_key in self.restrict_cache:
                restricted_df = self.restrict_cache[cache_key]
            else:
                restrict = True
                for var, layout in cache_key:
                    restrict = restrict & (self.data[var] == layout)
                restricted_df = self.data[restrict]
                self.restrict_cache[cache_key] = restricted_df
            if len(var_order) > cache_vars:
                restrict = True
                for var in var_order[cache_vars:]:
                    restrict = restrict & (restricted_df[var] == cfg[var])
                restricted_df = restricted_df[restrict]
        else:
            # There might be a better way to do this.
            # This iteratively builds up the matching indices.
            restrict = True
            for var, layout in cfg.items():
                if var in self.specials: continue
                restrict = restrict & (self.data[var] == layout)
            restricted_df = self.data[restrict]
        try:
            return self._get_row(restricted_df['time'].idxmin())
        except ValueError as e:
            print(f'Exception with restrict config {cfg} in {self.name}')
            raise e


    def get_layout_for_config(self, config, specials=False):
        """Return a layout dict for config.

        config should be the result of get_min_config or similar.

        If specials is True, also return special configurations.

        """
        layout = {}
        for var in self.inputs + self.outputs:
            layout[var] = config[var]
        if specials:
            for special in self.specials:
                layout[special] = config[special]
        return layout


    def _get_row(self, idx):
        """Return the row in data for an index."""
        return self.data.loc[idx]


    def _filter_fused_configs(self):
        """Remove fused QKV configs where the stacking is not the first dimension."""
        idx = set()
        for col in ['A', 'B', 'C']:
            for i, v in enumerate(self.data[col]):
                if 'q' not in v: break
                if v[0] != 'q': idx.add(i)
        self.data.drop(self.data.index[list(idx)], inplace=True)


    def _rename_aib_configs(self):
        """Replace j with k for some aib configs."""
        if self.name == 'aib':
            to_replace = ['wkk', 'wvv', 'kk', 'vv']
        elif self.name == 'baib':
            to_replace = ['dkk', 'dvv']
        for col in to_replace:
            self.data[col] = self.data[col].map(
                lambda x: x.replace('j', 'k'))


# Used by Split/MergeOperator for returning configs.
DummyEntry = namedtuple('DummyEntry', ['time', 'cfg'])


class SplitOperator:
    """Special operator for splitting QKV-fused output for aib.

    This also translates 'j' to 'k' for WKK and WVV.

    Note it should not be *too* hard to generalize this if needed.

    """

    def __init__(self, opdef, input_op):
        """Set up the split.

        input_op is the actual operator that will provide the input,
        which is needed so this operator can produce the right layouts.

        """
        self.name = opdef.name
        self.input_op = input_op
        self.inputs = opdef.input
        if len(self.inputs) != 1:
            raise ValueError(f'Split {self.name} does not support more than one input')
        self.outputs = opdef.output
        self.all_vars = set(self.inputs + self.outputs)
        self.specials = []

        # Precompute layouts.
        self.unique_layouts = dict(
            self.input_op.get_unique_layouts(self.inputs))
        for var in self.outputs:
            self.unique_layouts[var] = [
                x.replace('q', '') for x in
                self.unique_layouts[self.inputs[0]]]
            if var in ['WKK', 'WVV']:
                self.unique_layouts[var] = [
                    x.replace('j', 'k') for x in
                    self.unique_layouts[var]]


    def get_unique_layouts(self, vars=None, cfg=None):
        """Like Operator.get_unique_layouts"""
        layouts = self.unique_layouts
        if cfg:
            new_layouts = {var: [] for var in layouts.keys()}
            for layout in layout_iterator(layouts, cfg=cfg):
                for var, var_layout in layout.items():
                    new_layouts[var].append(var_layout)
            layouts = new_layouts
        if vars is not None:
            return {var: layouts[var] for var in vars}
        return layouts


    def get_min_config(self, cfg=None, cache=False):
        """Like Operator.get_min_config.

        This returns a "fake" configuration, not a real Pandas object.

        The relevant thing is that the time is always 0.

        """
        return DummyEntry(0.0, cfg)


    def get_layout_for_config(self, config, specials=False):
        """Like Operator.get_layout_for_config."""
        if config.cfg is None:
            return {}
        return config.cfg



class MergeOperator:
    """Special operator for merging variables for dX/WQKV-fused.

    Note this can probably be generalized if needed.

    """

    def __init__(self, opdef, input_ops):
        self.name = opdef.name
        self.input_ops = input_ops
        self.inputs = opdef.input
        self.outputs = opdef.output
        if len(self.outputs) != 1:
            raise ValueError(f'Merge {self.name} does not support more than one output')
        self.all_vars = set(self.inputs + self.outputs)
        self.specials = []

        # Precompute layouts.
        self.unique_layouts = {}
        for op in self.input_ops:
            for var in self.inputs:
                if var in op.outputs:
                    layouts = op.get_unique_layouts([var])[var]
                    # Temporarily translate all k to j.
                    layouts = [x.replace('k', 'j') for x in layouts]
                    # Different operators might report layouts in different
                    # orders, so this standardizes them.
                    layouts.sort()
                    self.unique_layouts[var] = layouts
        # Verify all inputs have the same layouts.
        check = [self.unique_layouts[var] for var in self.inputs]
        if check.count(check[0]) != len(check):
            raise ValueError(f'Some inputs to {self.name} do not have identical layouts')
        self.unique_layouts[self.outputs[0]] = [
            'q' + x for x in self.unique_layouts[self.inputs[0]]]
        # Retranslate j to k.
        for var in ['DKK', 'DVV']:
            self.unique_layouts[var] = [
                x.replace('j', 'k') for x in self.unique_layouts[var]]


    def get_unique_layouts(self, vars=None, cfg=None):
        """Like operator.get_unique_layouts."""
        layouts = self.unique_layouts
        if cfg:
            new_layouts = {var: [] for var in layouts.keys()}
            for layout in layout_iterator(layouts, cfg=cfg):
                for var, var_layout in layout.items():
                    new_layouts[var].append(var_layout)
            layouts = new_layouts
        if vars is not None:
            return {var: layouts[var] for var in vars}
        return layouts


    def get_min_config(self, cfg=None, cache=False):
        """Like Operator.get_min_config.

        This returns a "fake" configuration, not a real Pandas object.

        The relevant thing is that the time is always 0.

        """
        return DummyEntry(0.0, cfg)


    def get_layout_for_config(self, config, specials=False):
        """Like Operator.get_layout_for_config."""
        if config.cfg is None:
            return {}
        return config.cfg



class CombinedOperator:
    """Combine multiple operators together."""

    def __init__(self, ops):
        self.name = 'combined_' + '_'.join([op.name for op in ops])
        self.ops = ops
        # Deduplicate inputs/outputs.
        inputs, outputs = set(), set()
        for op in ops:
            inputs.update(op.inputs)
            outputs.update(op.outputs)
            
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.all_vars = set(self.inputs + self.outputs)
        self.specials = []


    def get_unique_layouts(self, vars=None, cfg=None):
        """Like Operator.get_unique_layouts."""
        # Get the unique layouts for each operator.
        layouts = []
        for op in self.ops:
            # Restrict vars to those relevant to this operator.
            op_vars = filter_layouts(vars, op.inputs + op.outputs)
            # Likewise cfg.
            op_cfg = filter_layouts(cfg, op.inputs + op.outputs)
            op_layouts = op.get_unique_layouts(op_vars, cfg=op_cfg)
            layouts.append(op_layouts)
        # Expand and merge layouts.
        return merge_layouts(layouts)


    def get_min_config(self, cfg=None, cache=False):
        """Like Operator.get_min_config.

        This returns a "fake" configuration, not a real Pandas object.

        The time is the sum of the minimum times for all operators.

        """
        t = 0.0
        op_configs = {}
        for op in self.ops:
            op_cfg = filter_layouts(cfg, op.inputs + op.outputs)
            op_config = op.get_min_config(op_cfg, cache=cache)
            op_configs[op.name] = op_config
            t += op_config.time
        return DummyEntry(t, op_configs)


    def get_layout_for_config(self, config, specials=False):
        """Like Operator.get_layout_for_config.

        Specials are handled specially.

        """
        layout = {}
        if specials:
            layout['specials'] = {}
        for op in self.ops:
            op_layout = op.get_layout_for_config(config.cfg[op.name], specials=specials)
            # Mostly doing this to sanity-check layouts.
            for var, var_layout in op_layout.items():
                if var in layout:
                    if layout[var] != var_layout:
                        raise RuntimeError(f'Got mismatched layouts in combined operator, {op.name} wants {var} = {var_layout} but current value is {layout[var]}')
                else:
                    if specials and var in op.specials:
                        layout['specials'][op.name] = {}
                        for special in op.specials:
                            layout['specials'][op.name][special] = op_layout[special]
                    else:
                        layout[var] = var_layout
        return layout


def load_operator(opdef, result_dir, loaded_ops):
    """Load an operator."""
    if opdef.type in ('tc', 'kernel'):
        return Operator(opdef, result_dir)
    elif opdef.type == 'split':
        return SplitOperator(
            opdef, loaded_ops[opdef.bm])
    elif opdef.type == 'merge':
        return MergeOperator(
            opdef, [loaded_ops[x] for x in opdef.bm])
    else:
        raise ValueError(f'Unknown operator type {opdef.type}')


def load_ops(result_dir):
    """Load operators using data in result_dir."""
    fwd_ops, bwd_ops = {}, {}
    for opdef in fwd_operators:
        op = load_operator(opdef, result_dir, fwd_ops)
        fwd_ops[op.name] = op
    for opdef in bwd_operators:
        op = load_operator(opdef, result_dir, bwd_ops)
        bwd_ops[op.name] = op
    return fwd_ops, bwd_ops



class Variables:
    """Manage information on variables used by operators."""

    def __init__(self, fwd_ops, bwd_ops, same_constraints,
                 fwd_merges=None, bwd_merges=None):
        """Use variables defined by forward/backward operators."""
        # Add operators, combining if needed.
        self.fwd_ops = {}
        removed_ops = set()
        if fwd_merges is not None:
            for merge in fwd_merges:
                removed_ops.update(merge)
                combined_op = CombinedOperator(
                    [fwd_ops[opname] for opname in merge])
                self.fwd_ops[combined_op.name] = combined_op
        for opname, op in fwd_ops.items():
            if opname not in removed_ops:
                self.fwd_ops[opname] = op
        self.bwd_ops = {}
        removed_ops = set()
        if bwd_merges is not None:
            for merge in bwd_merges:
                removed_ops.update(merge)
                combined_op = CombinedOperator(
                    [bwd_ops[opname] for opname in merge])
                self.bwd_ops[combined_op.name] = combined_op
        for opname, op in bwd_ops.items():
            if opname not in removed_ops:
                self.bwd_ops[opname] = op
        self.ops = dict(self.fwd_ops)
        self.ops.update(self.bwd_ops)
        # Also useful to directly have the unmerged versions.
        self.unmerged_ops = dict(fwd_ops)
        self.unmerged_ops.update(bwd_ops)

        self.same_constraints = same_constraints

        # Variables used in forward/backward prop.
        self.fwd_vars, self.bwd_vars = set(), set()
        for op in fwd_ops.values():
            self.fwd_vars.update(op.inputs)
            self.fwd_vars.update(op.outputs)
        for op in bwd_ops.values():
            self.bwd_vars.update(op.inputs)
            self.bwd_vars.update(op.outputs)

        # Build a map from variables to the operators that use them.
        self.vars_to_ops = defaultdict(list)
        self.var_producers = defaultdict(list)
        self.var_consumers = defaultdict(list)
        for name, op in self.ops.items():
            for var in op.inputs:
                self.vars_to_ops[var].append(op)
                self.var_consumers[var].append(op)
            for var in op.outputs:
                self.vars_to_ops[var].append(op)
                self.var_producers[var].append(op)

        # Map from each variable to all variables it must be the same as
        # (including itself).
        same_map = {}
        for var in self.vars_to_ops.keys():
            same_map[var] = set([var])
            for constraint in self.same_constraints:
                if var in constraint:
                    for var2 in constraint:
                        if var2 not in self.vars_to_ops:
                            raise ValueError(f'Unknown variable {var2} in same constraint for {var}')
                    same_map[var] |= constraint
        # Now iterate until this converges to propagate all sameness.
        old_same_map = dict(same_map)
        while True:
            for var, same_set in same_map.items():
                for same_var in same_set:
                    same_map[same_var] |= same_set
            if same_map == old_same_map:
                break
            old_same_map = dict(same_map)
        self.same_map = same_map

        # Precompute layouts for operators and variables.
        self._precompute_layouts()

        # Sanity check that every variable has at least one layout.
        for var in self.vars_to_ops.keys():
            layouts = self.get_layouts_for_var(var)
            if not layouts:
                print(f'WARNING: Variable {var} has no valid layouts!')

        # Sanity check that every operator has at least one possible layout.
        for opname in self.ops:
            t = time.perf_counter()
            layouts = self.get_valid_unique_layouts(opname)
            if not next(iter(layouts.values())):
                print(f'WARNING: Operator {opname} has no valid configurations!')


    def print_vars(self):
        """Print information on variables."""
        for var in self.vars_to_ops.keys():
            print(f'''{var}:
\tProducers: {[op.name for op in self.var_producers[var]]}
\tConsumers: {[op.name for op in self.var_consumers[var]]}
\tSame as: {self.same_map[var]}
\tPotential layouts: {self.get_layouts_for_var(var)}''')


    def print_op_info(self):
        """Print information on operators."""
        for opname in self.ops:
            layouts = self.get_valid_unique_layouts(opname)
            num_layouts = len(next(iter(layouts.values())))
            print(f'{opname}: {num_layouts} total layouts')


    def vars_between(self, op1, op2):
        """Return a list of variables that are produced by op1 and
        consumed by op2."""
        return set(op1.outputs) & set(op2.inputs)


    @functools.lru_cache(maxsize=None)
    def get_valid_unique_layouts(self, opname, vars=None, binding=None):
        """Return valid unique layouts for operator.

        This is like Operator.get_unique_layouts, except it will
        take into account constraints on variables.

        Specifically, a layout is valid if:
        - It works with currently-bound variables.
        - The layouts of variables permit at least one layout for all
        operators that use the variables.
        - The prior constraint is also satisfied for all variables that
        must be the same as the variables used by the operator.

        This *should* account for propagating constraints on variables.
        That is, picking a layout for one variable might constrain some
        operator such that the output layouts it can produce now result
        in some other operator having constraints on layouts.

        """
        if binding:
            # Only have to recompute if we restrict layouts further.
            # Turn back into a dict.
            frozen_binding = binding
            binding = dict(binding)
            # First check if anything in the binding is set for the ops.
            if not any([self.get_op_config_from_binding(op, binding) for
                        op in self.compatible_op_layouts]):
                layouts = self.compatible_op_layouts[opname]
            else:
                # Check if we cached this.
                if frozen_binding in self.compatible_op_layouts_cache:
                    compatible_layouts = self.compatible_op_layouts_cache[frozen_binding]
                else:
                    # Use precomputed layouts, restricted by the binding.
                    layouts = [
                        restrict_layout_configs(
                            op_layout, self.get_op_config_from_binding(op, binding))
                        for op, op_layout in self.compatible_op_layouts.items()]
                    # Reapply this to propagate any further restrictions.
                    compatible_layouts = get_compatible_layouts(layouts, self.same_map)
                    # Cache.
                    self.compatible_op_layouts_cache[frozen_binding] = compatible_layouts
                op_idx = list(self.compatible_op_layouts.keys()).index(opname)
                layouts = compatible_layouts[op_idx]
        else:
            layouts = self.compatible_op_layouts[opname]
        # Now restrict the layout by vars and unique'ify.
        if vars is None:
            vars = self.ops[opname].inputs + self.ops[opname].outputs
        unique_layouts_set = set(bare_layout_iterator(layouts, order=vars))
        unique_layouts = {var: [] for var in vars}
        for bare_layout in unique_layouts_set:
            for var, layout in zip(vars, bare_layout):
                unique_layouts[var].append(layout)
        return unique_layouts


    def get_layouts_for_var(self, var, binding=None):
        """Return the set of possible layouts for var.

        This accounts for possible restrictions on the variable's layout
        due to sameness or what operators accept.

        """
        if binding:
            if binding[var] is not None:
                # Only one possible layout: The one you already picked.
                return set([binding[var]])
            # Just get the set of unique layouts for an operator that uses var.
            layouts = self.get_valid_unique_layouts(
                self.vars_to_ops[var][0].name, vars=(var,),
                binding=freeze_dict(binding))
            return set(layouts[var])
        else:
            return self.compatible_var_layouts[var]


    def empty_var_binding(self):
        """Return an empty variable layout binding."""
        return {var: None for var in self.vars_to_ops.keys()}


    def set_var_binding(self, binding, var, layout):
        """Bind a variable.

        This will set the layout for the variable and any other variables that
        must be the same.

        This will throw an error if any of them are already bound and have a
        different layout.

        """
        new_binding = dict(binding)
        for same_var in self.same_map[var]:
            if binding[same_var] is not None and binding[same_var] != layout:
                raise RuntimeError(f'Tried to bind {var} to {layout} but {same_var} already has layout {binding[same_var]}')
            new_binding[same_var] = layout
        return new_binding


    def update_binding_from_cfg(self, binding, cfg):
        """Update a binding based on a configuration.

        This will bind all variables in cfg to their given layouts.

        """
        new_binding = dict(binding)
        for var, layout in cfg.items():
            for same_var in self.same_map[var]:
                if binding[same_var] is not None and binding[same_var] != layout:
                    raise RuntimeError(f'Tried to bind {var} to {layout} but {same_var} already has layout {new_binding[same_var]}')
                new_binding[same_var] = layout
        return new_binding


    def get_op_config_from_binding(self, opname, binding, hashable=False):
        """Return a cfg dict (for get_unique_layouts) for opname based on the
        currently bound variables in binding.

        If no relevant variables are bound, this returns None.
        If only some are bound, it only restricts them.

        If hashable is True, this will return a hashable "dict".

        """
        if not binding:
            return None
        cfg = {}
        for var in self.ops[opname].inputs + self.ops[opname].outputs:
            if binding[var] is not None:
                cfg[var] = binding[var]
        if cfg:
            if hashable:
                return freeze_dict(cfg)
            else:
                return cfg
        else:
            return None


    def minimize_binding(self, binding):
        """Set unbound variable layouts in binding to minimize time."""
        # Handle trivial bindings where the variable has only one layout.
        # Iterate on this in case binding variables reduces layouts on others.
        while True:
            bound_something = False
            for var in binding:
                if binding[var] is not None: continue
                var_layouts = self.get_layouts_for_var(var, binding)
                if len(var_layouts) == 1:
                    binding[var] = var_layouts.pop()
                    bound_something = True
            if not bound_something:
                break

        for var in list(binding.keys()):
            if binding[var] is not None: continue
            # Find the set of operators influenced by var directly or through
            # other unbound variables used by the same operators.
            vars_to_bind, ops_involved = self._get_influenced_ops_and_vars(
                var, binding)
            # It tends to get too expensive to minimize exhaustively over too
            # many operators, so skip over to the greedy approach.
            while len(ops_involved) > 3:
                print(f'Too many ops {", ".join([op.name for op in ops_involved])} '
                      f'(need to bind {", ".join(vars_to_bind)}), switching to greedy')
                binding = self._greedy_bind_one_op(ops_involved, binding)
                vars_to_bind, ops_involved = self._get_influenced_ops_and_vars(
                    var, binding)
            if not vars_to_bind:
                continue  # var got bound greedily.
            # Now do exhaustive minimization over all possible layouts.
            print(f'Exhaustively minimizing over {", ".join([op.name for op in ops_involved])}')
            min_layout = self._exhaustively_minimize_ops(ops_involved, binding)
            print(f'Exhaustively minimized {", ".join([op.name for op in ops_involved])}, setting',
                  ' '.join([f'{var}={layout}' for var, layout in min_layout.items()]))
            # Now bind the variables.
            for unbound_var in vars_to_bind:
                binding[unbound_var] = min_layout[unbound_var]

        return binding


    def check_binding(self, binding):
        """Check that a binding is actually feasible.

        This will throw an exception if it does not work.

        """
        for op in self.ops.values():
            cfg = filter_layouts(binding, op.inputs + op.outputs)
            op.get_min_config(cfg)


    def get_operator_configs_for_binding(self, binding):
        """Return layouts, with specials, for each operator based on binding.

        Specials will be determined by finding the minimum configuration for
        each operator.

        This will "unpack" combined operators for simplicity.

        """
        configs = {}
        for opname, op in self.unmerged_ops.items():
            layout = filter_layouts(binding, op.inputs + op.outputs)
            config = op.get_min_config(layout)
            configs[opname] = op.get_layout_for_config(config, specials=True)
        return configs


    def _precompute_layouts(self):
        """Precompute valid layouts for variables and operators."""
        op_order = list(self.ops.keys())
        layouts = [self.ops[opname].get_unique_layouts() for opname in op_order]
        compatible_layouts = get_compatible_layouts(layouts, self.same_map)
        self.compatible_op_layouts = {opname: layout for opname, layout in zip(
            op_order, compatible_layouts)}
        for opname, layouts in self.compatible_op_layouts.items():
            if not layout_len(layouts):
                raise RuntimeError(f'No compatible layouts for {opname}')
        self.compatible_var_layouts = {}
        for var, ops in self.vars_to_ops.items():
            self.compatible_var_layouts[var] = set(
                self.compatible_op_layouts[ops[0].name][var])
            if not self.compatible_var_layouts[var]:
                raise RuntimeError(f'No compatible layouts for {var}')
            # Sanity check.
            for op in ops[1:]:
                if self.compatible_var_layouts[var] != set(
                        self.compatible_op_layouts[op.name][var]):
                    raise RuntimeError(f'Compatible layouts for {var} do not match across ops')
        self.compatible_op_layouts_cache = {}


    def _get_influenced_ops_and_vars(self, var, binding):
        """Get variables and operators influenced by binding var.

        If var is bound, this returns empty sets.

        """
        if binding[var]:
            return set(), set()
        vars_to_bind = set([var])
        ops_involved = set()
        old_vars_to_bind = set(vars_to_bind)
        while True:
            # Add ops that use vars.
            for unbound_var in vars_to_bind:
                ops_involved.update(self.vars_to_ops[unbound_var])
            # Add new unbound vars from the ops.
            for op in ops_involved:
                for op_var in op.inputs + op.outputs:
                    # Only need to check sameness here:
                    # If it were bound, all vars that are the same are bound.
                    if binding[op_var] is None:
                        for same_var in self.same_map[op_var]:
                            vars_to_bind.add(same_var)
            if vars_to_bind == old_vars_to_bind:
                break
            old_vars_to_bind = set(vars_to_bind)
        return vars_to_bind, ops_involved

    def _exhaustively_minimize_ops(self, ops_involved, binding):
        """Exhaustively search for the minimal binding of ops."""
        layouts = [self.get_valid_unique_layouts(
            op.name, binding=freeze_dict(binding)) for op in ops_involved]
        layouts = merge_layouts(layouts, sameness=self.same_map)
        print(f'Considering {layout_len(layouts)} layouts...')
        min_t = float('inf')
        min_layout = None
        for layout in layout_iterator(layouts):
            total_t = 0.0
            for op in ops_involved:
                op_cfg = filter_layouts(layout, op.inputs + op.outputs)
                total_t += op.get_min_config(cfg=op_cfg, cache=True).time
            if total_t < min_t:
                min_t = total_t
                min_layout = layout
        return min_layout


    def _greedy_bind_one_op(self, ops_involved, binding):
        """Pick a binding for one operator in ops.

        This is meant to reduce the number of involved operators/variables
        to make exhaustive minimization feasible later.

        Right now we pick the minimum layout for the most expensive operator.

        """
        min_cfgs = {op.name: op.get_min_config(
            cfg=self.get_op_config_from_binding(op.name, binding))
                    for op in ops_involved}
        # Now pick the most expensive operator.
        max_op = max(min_cfgs, key=lambda x: min_cfgs[x].time)
        min_layout = self.ops[max_op].get_layout_for_config(min_cfgs[max_op])
        print(f'Greedily binding {max_op}, setting',
              ' '.join([f'{var}={layout}' for var, layout in min_layout.items()]))
        for var, layout in min_layout.items():
            if binding[var] is None:
                binding = self.set_var_binding(binding, var, layout)
            else:
                # Sanity check.
                assert binding[var] == layout
        return binding
