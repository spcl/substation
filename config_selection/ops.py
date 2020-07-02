import os.path
import functools
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
    # Dropout mask and tensor must match:
    ('ATTN_DROP_MASK', 'ATTN_DROP'),
    # These need to match because they are stacked:
    # Removed because of issues with j vs k.
    #('DQQ', 'DKK', 'DVV')
    # Presently, a hack:
    ('DKKQQVV', 'WKKWQQWVV')
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
        cfg = {}
    vars = list(restrict) if restrict else list(layouts.keys())
    num_layouts = len(layouts[vars[0]])
    for i in range(num_layouts):
        match = True
        for var, layout in cfg.items():
            if layouts[var][i] != layout:
                match = False
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
    if type(x) in (list, tuple):
        x = [var for var in x if var in vars]
    elif type(x) is dict:
        x = {var: x[var] for var in x if var in vars}
    else:
        raise ValueError(f'Do not know how to filter {x}')
    if not x:
        return None
    else:
        return x


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

        # Set up inputs/output details.
        # inputs/output are a list of input/output variables.
        self.inputs, self.outputs = [], []
        # Maps variables to the corresponding dataframe columns and vice-versa.
        # In general, this is to a single column but one variable can
        # be used for multiple columns (occurs with bdrln1).
        # These cases are skipped for simplicity.
        self.var_map = {}
        self.df_to_var = {}
        for (var, data_var) in opdef.input:
            if data_var not in self.data.columns:
                raise ValueError(f'Input var {var}->{data_var} not in data for {self.name}')
            self.inputs.append(var)
            if var in self.var_map:
                #self.var_map[var].append(data_var)
                continue
            else:
                self.var_map[var] = [data_var]
            if data_var in self.df_to_var:
                #self.df_to_var[data_var].append(var)
                continue
            else:
                self.df_to_var[data_var] = [var]
        for (var, data_var) in opdef.output:
            if data_var not in self.data.columns:
                raise ValueError(f'Output var {var}->{data_var} not in data for {self.name}')
            self.outputs.append(var)
            if var in self.var_map:
                #self.var_map[var].append(data_var)
                continue
            else:
                self.var_map[var] = [data_var]
            if data_var in self.df_to_var:
                #self.df_to_var[data_var].append(var)
                continue
            else:
                self.df_to_var[data_var] = [var]

        self.specials = []
        self.specials_map = {}  # Map specials to dataframe columns.
        if opdef.special:
            for special in opdef.special:
                # For simplicity, since most do not change.
                if type(special) is tuple:
                    self.specials_map[special[0]] = special[1]
                else:
                    self.specials_map[special] = special
        for special, specialdf in self.specials_map.items():
            self.specials.append(special)
            if specialdf not in self.data.columns:
                raise ValueError(f'Special {special}->{specialdf} not in data for {self.name}')


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
                for dfvar in self._vars2df(var):
                    restrict = restrict & (self.data[dfvar] == layout)
            df = self.data[restrict]
        else:
            df = self.data
        df = df.drop_duplicates(self._vars2df(vars))
        cfgs = {}
        for var in vars:
            cfgs[var] = list()
            for dfvar in self._vars2df(var):
                cfgs[var] += list(df[dfvar])
        return cfgs


    def get_min_config(self, cfg=None):
        """Return the configuration with minimum time.

        cfg can be used to optionally restrict the minimum to be only over
        vars with a particular layout. It should be a dict mapping vars to
        the bound layout.

        """
        if cfg is None:
            return self._get_row(self.data['time'].idxmin())

        # There might be a better way to do this.
        # This iteratively builds up the matching indices.
        restrict = True
        for var, layout in cfg.items():
            if var in self.specials: continue
            for dfvar in self._vars2df(var):
                restrict = restrict & (self.data[dfvar] == layout)
        try:
            return self._get_row(self.data[restrict]['time'].idxmin())
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
            for dfvar in self._vars2df(var):
                layout[var] = config[dfvar]
        if specials:
            for special, specialdf in self.specials_map.items():
                layout[special] = config[specialdf]
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


    def _vars2df(self, vars):
        """Translate every var in vars to the dataframe column."""
        if type(vars) not in (list, tuple):
            vars = [vars]
        mapped = []
        for var in vars:
            if var not in self.var_map:
                raise ValueError(f'Unknown variable {var} for {self.name}')
            mapped += self.var_map[var]
        return mapped


    def _df2vars(self, vars):
        """Translate every var in vars to the regular variable name."""
        if type(vars) not in (list, tuple):
            vars = [vars]
        mapped = []
        for var in vars:
            if var not in self.df_to_var:
                raise ValueError(f'Unknown variable {var} for {self.name}')
            mapped += self.df_to_var[var]
        return mapped


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


    def get_min_config(self, cfg=None):
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


    def get_min_config(self, cfg=None):
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
        new_layouts = layouts[0]
        for op_layout in layouts[1:]:
            new_layouts = self._expand_layouts(new_layouts, op_layout)
        return new_layouts


    def get_min_config(self, cfg=None):
        """Like Operator.get_min_config.

        This returns a "fake" configuration, not a real Pandas object.

        The time is the sum of the minimum times for all operators.

        """
        t = 0.0
        op_configs = {}
        for op in self.ops:
            op_cfg = filter_layouts(cfg, op.inputs + op.outputs)
            op_config = op.get_min_config(op_cfg)
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


    def _expand_layouts_no_shared(self, layouts1, layouts2):
        """Merge two sets of layouts, when they have no variables in common."""
        # Essentially, for each layout in layouts1, this will add every
        # layout in layouts2.
        # Verify no variables in common:
        assert not (set(layouts1.keys()) & set(layouts2.keys()))
        new_layouts = {var: [] for var in list(layouts1.keys()) + list(layouts2.keys())}
        for layout1 in layout_iterator(layouts1):
            for layout2 in layout_iterator(layouts2):
                for var, layout in layout1.items():
                    new_layouts[var].append(layout)
                for var, layout in layout2.items():
                    new_layouts[var].append(layout)
        return new_layouts


    def _expand_layouts(self, layouts1, layouts2):
        """Merge two sets of layouts, which may have variables in common.

        If there are incompatible layouts of common variables, they are dropped.

        """
        # If there are none in common, use the other function.
        common_vars = set(layouts1.keys()) & set(layouts2.keys())
        if not common_vars:
            return self._expand_layouts_no_shared(layouts1, layouts2)
        # We first find the unique set of layouts in common between the two
        # among the shared variables.
        shared_layouts1, shared_layouts2 = set(), set()
        for layout in layout_iterator(layouts1, restrict=common_vars):
            layout = tuple(sorted(layout.items()))
            shared_layouts1.add(layout)
        for layout in layout_iterator(layouts2, restrict=common_vars):
            layout = tuple(sorted(layout.items()))
            shared_layouts2.add(layout)
        shared_layouts = shared_layouts1 & shared_layouts2
        if not shared_layouts:
            raise RuntimeError('No common layouts to expand.')
        # Then we build a map from those to *every* layout in layouts2 that
        # matches this layout.
        layout_map = {layout: [] for layout in shared_layouts}
        for layout in layout_iterator(layouts2):
            shared_layout = {var: layout[var] for var in common_vars}
            shared_layout = tuple(sorted(shared_layout.items()))
            if shared_layout in layout_map:
                layout_map[shared_layout].append(layout)
        # Finally, for every layout in layouts1, we add every layout in layouts2
        # where the shared variables match.
        new_layouts = {var: [] for var in list(layouts1.keys()) + list(layouts2.keys())}
        for layout1 in layout_iterator(layouts1):
            shared_layout = {var: layout1[var] for var in common_vars}
            shared_layout = tuple(sorted(shared_layout.items()))
            if shared_layout not in layout_map:
                # Not fully sure when this happens, but possibly with
                # configuration restrictions.
                continue
            for layout2 in layout_map[shared_layout]:
                for var, layout in layout1.items():
                    new_layouts[var].append(layout)
                for var, layout in layout2.items():
                    if var not in common_vars:  # Don't add again.
                        new_layouts[var].append(layout)
        return new_layouts


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
        old_same_map = same_map
        while True:
            for var, same_set in same_map.items():
                for same_var in same_set:
                    same_map[same_var] |= same_set
            if same_map == old_same_map:
                break
            old_same_map = same_map
        self.same_map = same_map

        # Sanity check that every variable has at least one layout.
        for var in self.vars_to_ops.keys():
            layouts = self.get_layouts_for_var(var)
            if not layouts:
                print(f'WARNING: Variable {var} has no valid layouts!')

        # Sanity check that every operator has at least one possible layout.
        for opname in self.ops:
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
    def get_valid_unique_layouts(self, opname, vars=None, cfg=None):
        """Return unique layouts for operator.

        This is the same as Operator.get_unique_layouts except it will
        take into account constraints (sameness, layouts supported by
        other operators).

        Specifically, this will eliminate layouts if they are used for
        a variable and:
        - The variable is used by another op that does not support that
        layout.
        - The variable must be the same as another variable, and that
        variable is used by an op that does not support the layout.

        Note this does not deal with propagating constraints on variables,
        which is a TODO.

        This also will not account for *joint* layouts between variables:
        Some operators cannot produce some layouts for two variables
        simultaneously.

        """
        if cfg is not None:
            # Because of caching, need to convert back to dict.
            regular_cfg = dict(cfg)
        else:
            regular_cfg = None
        layouts = self.ops[opname].get_unique_layouts(vars, cfg=regular_cfg)
        # Note that we are not concerned about the order of layouts here.
        # Essentially we filter out layouts that other operators do not accept.
        layout_sets = {var: self.get_layouts_for_var(var, cfg=cfg) for var in layouts}
        # Now filter the layouts.
        indices_to_remove = set()
        for var in layouts.keys():
            for i, layout in enumerate(layouts[var]):
                if layout not in layout_sets[var]:
                    indices_to_remove.add(i)
        filtered_layouts = {}
        for var in layouts.keys():
            filtered_layouts[var] = [x for i, x in enumerate(layouts[var])
                                     if i not in indices_to_remove]
        return filtered_layouts


    def get_valid_unique_layouts2(self, opname, vars=None, cfg=None, binding=None):
        """Return valid unique layouts for operator.

        This is like Operator.get_unique_layouts, except it will
        take into account constraints on variables.

        Specifically, a layout is valid if:
        - It works with currently-bound variables.
        - The layouts of variables permit at least one layout for all
        operators that use the variables.
        - The prior constraint is also satisfied for all variables that
        must be the same as the variables used by the operator.

        Note this does not propagate constraints on variables. That is,
        this considers only "first-order" constraints on operators that
        directly use variables. It does not, for example, consider whether
        chosing a layout for one variable might constrain other operators
        such that they have no possible layouts. (For example: The choice
        of an output variable for one operator may mean that the possible
        output layouts of a second operator are constrained such that a
        third operator now has no valid layouts.)

        """
        pass


    @functools.lru_cache(maxsize=None)
    def get_layouts_for_var(self, var, cfg=None):
        """Return the set of possible layouts for var.

        This accounts for what an operator can output as well as
        sameness constraints.

        """
        if cfg is not None:
            cfg = dict(cfg)
        layouts = []
        for same_var in self.same_map[var]:
            for op in self.vars_to_ops[same_var]:
                # Only pass along relevant variables.
                op_cfg = filter_layouts(cfg, op.inputs + op.outputs)
                layout_set = set(op.get_unique_layouts([same_var], cfg=op_cfg)[same_var])
                layouts.append(layout_set)
        valid_layouts = layouts[0]
        for layout_set in layouts[1:]:
            valid_layouts &= layout_set
        return valid_layouts


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


    def get_op_config_from_binding(self, opname, binding, hashable=False):
        """Return a cfg dict (for get_unique_layouts) for opname based on the
        currently bound variables in binding.

        If no relevant variables are bound, this returns None.
        If only some are bound, it only restricts them.

        If hashable is True, this will return a hashable "dict".

        """
        cfg = {}
        for var in self.ops[opname].inputs + self.ops[opname].outputs:
            if binding[var] is not None:
                cfg[var] = binding[var]
        if cfg:
            if hashable:
                return tuple(sorted(cfg.items()))
            else:
                return cfg
        else:
            return None


    def minimize_binding(self, binding):
        """Set unbound variable layouts in binding to minimize time."""
        # This is a simple greedy approach. Better would be to minimize over
        # all operators that use the variable.
        for var in list(binding.keys()):
            if binding[var] is not None: continue
            # Just pick the first operator to use for this.
            op = self.vars_to_ops[var][0]
            bound_vars, unbound_vars = {}, []
            for op_var in op.inputs + op.outputs:
                if binding[op_var] is None:
                    unbound_vars.append(op_var)
                else:
                    bound_vars[op_var] = binding[op_var]
            # Find the minimum config with the bound variables.
            cfg = op.get_min_config(bound_vars)
            cfg_layout = op.get_layout_for_config(cfg)
            for unbound_var in unbound_vars:
                binding[unbound_var] = cfg_layout[unbound_var]
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


# Try binding variables from FP ops before doing backprop optimization
# Try optimizing backprop before forward prop
# Improve minimize_binding to consider constraints on all operators that use a variable
# Improve minimize_binding to minimize over all operators that use the variable
