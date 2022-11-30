from math import log10, floor
from typing import Any

from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
from prettytable import PrettyTable

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


mpl_cmaps = {x: plt.get_cmap(x) for x in ['viridis', 'plasma', 'inferno', 'magma', 'cividis']}
cmaps = { # http://colormind.io/
    'nice': ['#0051a2', '#97964a', '#ffd44f', '#f4777f', '#93003a']
} | mpl_cmaps

def plot_setup(
    fsize   = 15,
    tsize   = 18, 
    tdir    = 'in',
    major   = 5.0,
    minor   = 3.0,
    lwidth  = 0.8,
    lhandle = 2.0,
    usetex  = True,
    style   = 'default',
    figsize = (5, 5),
    figshape= (1, 1),
):
    plt.style.use(style)

    plt.rcParams['text.usetex'] = usetex
    plt.rcParams['font.size'] = fsize
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    plt.rcParams['axes.linewidth'] = lwidth

    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = 5.0
    plt.rcParams['ytick.minor.size'] = 3.0

    plt.rcParams['legend.fontsize'] = tsize
    plt.rcParams['legend.handlelength'] = lhandle

    fig, axs = plt.subplots(*figshape, figsize=(figshape[1]*figsize[1], figshape[0]*figsize[0]))

    return fig, axs

def plot_format(
    ax,
    aspect = None,

    title = '',
    xlabel = '',
    ylabel = '',

    titlepad = 8,
    xlabelpad = 8,
    ylabelpad = 8,

    xlim = None,
    xticks = None,
    xticklabels = None,

    ylim = None, 
    yticks = None, 
    yticklabels = None,


):  
    ax.set_title(title, pad=titlepad)
    ax.set_xlabel(xlabel, labelpad = xlabelpad)
    ax.set_ylabel(ylabel, labelpad = ylabelpad)

    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if xticklabels is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45)
    if yticklabels is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, rotation=45)

    if aspect is not None:
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*aspect)

def useful_instructions(ax):
    ax.tick_params(
        axis='both', 
        which='minor',
        bottom=False,
    )

    ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 3), useMathText=True, useOffset=True)
    ax.tick_params(axis='both', pad=3.)  # pad default is 4.

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_major_formatter('{x:.0f}')

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # cmap = get_cmap('viridis', iterable=False)
    # cmap = plt.get_cmap('viridis')
    # cmap = iter([cmap(x) for x in np.linspace(0, 1, n_plots)])

def format_val_to_csv(v: Any):
    return str(v)

def get_cmap(name: str, iterable: bool = True):
    return iter(cmaps[name]) if iterable else cmaps[name]
    
def pretty_table(
    df: pd.DataFrame, 
    cols: list = None,
    top: float = 100.0, 
    bottom: float = 0.01
):
    
    table = PrettyTable()
    for c in cols if not cols is None else df.columns:
        if df[c].dtypes not in [str, object]:
            if (top > df[c]).all() and (df[c] > bottom).all():
                df[c] = df[c].map(lambda x: '{:.3f}'.format(x)) 
            else:
                df[c] = df[c].map(lambda x: '{:.3e}'.format(x))
            
            table.add_column(c, list(df[c]))

### LATEX ###

def gen_latex_table_rows(
    d           : dict, 
    norm_table  : pd.DataFrame, 
    cols        : list, 
    fname       : str,
    cmap        = mpl_cmaps['plasma'],
):
    header = '&'.join(cols) + ' \\\\ \n'
    with open(fname, "w") as f:
        f.write(header)
        for fg in d['Functional Groups']:
            row = [fg]
            for col in cols[1:]:
                val = d[fg][col]
                std_name = col + '_std'
                prec = 2
                if std_name in d[fg].keys():
                    std = d[fg][std_name]
                    # prec = get_precision(std)
                    err = f'({str(std).lstrip("0.")[prec]})'
                else:
                    err = ''
                    print(val/1.1*norm_table[col])
                    rbg = np.array(cmap(val/(1.1*norm_table[col])))[:3]
                    rgb = ','.join([f"{x:.2f}".lstrip('0').strip(' ') for x in rbg])
                
                if not col == 'Proportion':
                    row.append((f'\cellcolor[rgb]{{{rgb}}} $' + str(val)[:2+prec] + str(err) + '$').strip())
                else:
                    row.append(f'${str(val)[:2+prec]}$')
            
            f.write(' & '.join(row) + ' \\\\ \n')


### PANDAS ###

def df_to_dict(df: pd.DataFrame, type_filter: list = []) -> dict:
    return {c:np.array(df[c]) for c in df.columns if df[c].dtypes not in type_filter}


### utils ###

def get_precision(x):
  prec = -int(floor(log10(abs(x))))
  return prec

def mkdir(path: Path) -> Path:
    path = Path(path)
    if path.suffix != '':
        path = path.parent
    if path.exists():
        print('path exists, leaving alone')
    else:
        path.mkdir(parents=True)
    return path

''' LINE PROPERTIES 
Line setters
  agg_filter: a filter function, which takes a (m, n, 3) float array and a dpi value, and returns a (m, n, 3) array and two offsets from the bottom left corner of the image
  alpha: scalar or None
  animated: bool
  antialiased or aa: bool
  clip_box: `.Bbox`
  clip_on: bool
  clip_path: Patch or (Path, Transform) or None
  color or c: color
  dash_capstyle: `.CapStyle` or {'butt', 'projecting', 'round'}
  dash_joinstyle: `.JoinStyle` or {'miter', 'round', 'bevel'}
  dashes: sequence of floats (on/off ink in points) or (None, None)
  data: (2, N) array or two 1D arrays
  drawstyle or ds: {'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'}, default: 'default'
  figure: `.Figure`
  fillstyle: {'full', 'left', 'right', 'bottom', 'top', 'none'}
  gapcolor: color or None
  gid: str
  in_layout: bool
  label: object
  linestyle or ls: {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
  linewidth or lw: float
  marker: marker style string, `~.path.Path` or `~.markers.MarkerStyle`
  markeredgecolor or mec: color
  markeredgewidth or mew: float
  markerfacecolor or mfc: color
  markerfacecoloralt or mfcalt: color
  markersize or ms: float
  markevery: None or int or (int, int) or slice or list[int] or float or (float, float) or list[bool]
  mouseover: bool
  path_effects: `.AbstractPathEffect`
  picker: float or callable[[Artist, Event], tuple[bool, dict]]
  pickradius: unknown
  rasterized: bool
  sketch_params: (scale: float, length: float, randomness: float)
  snap: bool or None
  solid_capstyle: `.CapStyle` or {'butt', 'projecting', 'round'}
  solid_joinstyle: `.JoinStyle` or {'miter', 'round', 'bevel'}
  transform: `.Transform`
  url: str
  visible: bool
  xdata: 1D array
  ydata: 1D array
  zorder: float
Line getters
    agg_filter = None
    alpha = None
    animated = False
    antialiased or aa = True
    bbox = Bbox(x0=0.0, y0=-1.0, x1=0.99, y1=1.0)
    children = []
    clip_box = TransformedBbox(     Bbox(x0=0.0, y0=0.0, x1=1.0, ...
    clip_on = True
    clip_path = None
    color or c = r
    dash_capstyle = butt
    dash_joinstyle = round
    data = (array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, ...
    drawstyle or ds = default
    figure = Figure(640x480)
    fillstyle = full
    gapcolor = None
    gid = None
    in_layout = True
    label = _child0
    linestyle or ls = --
    linewidth or lw = 2.0
    marker = None
    markeredgecolor or mec = r
    markeredgewidth or mew = 1.0
    markerfacecolor or mfc = r
    markerfacecoloralt or mfcalt = none
    markersize or ms = 6.0
    markevery = None
    mouseover = False
    path = Path(array([[ 0.00000000e+00,  0.00000000e+00],   ...
    path_effects = []
    picker = None
    pickradius = 5
    rasterized = False
    sketch_params = None
    snap = None
    solid_capstyle = projecting
    solid_joinstyle = round
    tightbbox = Bbox(x0=80.0, y0=52.8, x1=571.04, y1=422.4)
    transform = CompositeGenericTransform(     TransformWrapper(  ...
    transformed_clip_path_and_affine = (None, None)
    url = None
    visible = True
    window_extent = Bbox(x0=80.0, y0=-316.79999999999995, x1=571.04, y...
    xdata = [0.   0.01 0.02 0.03 0.04 0.05]...
    xydata = [[0.         0.        ]  [0.01       0.06279052] ...
    ydata = [0.         0.06279052 0.12533323 0.18738131 0.248...
    zorder = 2
Rectangle setters
  agg_filter: a filter function, which takes a (m, n, 3) float array and a dpi value, and returns a (m, n, 3) array and two offsets from the bottom left corner of the image
  alpha: scalar or None
  angle: unknown
  animated: bool
  antialiased or aa: bool or None
  bounds: (left, bottom, width, height)
  capstyle: `.CapStyle` or {'butt', 'projecting', 'round'}
  clip_box: `.Bbox`
  clip_on: bool
  clip_path: Patch or (Path, Transform) or None
  color: color
  edgecolor or ec: color or None
  facecolor or fc: color or None
  figure: `.Figure`
  fill: bool
  gid: str
  hatch: {'/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
  height: unknown
  in_layout: bool
  joinstyle: `.JoinStyle` or {'miter', 'round', 'bevel'}
  label: object
  linestyle or ls: {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
  linewidth or lw: float or None
  mouseover: bool
  path_effects: `.AbstractPathEffect`
  picker: None or bool or float or callable
  rasterized: bool
  sketch_params: (scale: float, length: float, randomness: float)
  snap: bool or None
  transform: `.Transform`
  url: str
  visible: bool
  width: unknown
  x: unknown
  xy: (float, float)
  y: unknown
  zorder: float
Rectangle getters
    agg_filter = None
    alpha = None
    angle = 0.0
    animated = False
    antialiased or aa = True
    bbox = Bbox(x0=0.0, y0=0.0, x1=1.0, y1=1.0)
    capstyle = butt
    center = [0.5 0.5]
    children = []
    clip_box = None
    clip_on = True
    clip_path = None
    corners = [[0. 0.]  [1. 0.]  [1. 1.]  [0. 1.]]
    data_transform = BboxTransformTo(     TransformedBbox(         Bbox...
    edgecolor or ec = (0.0, 0.0, 0.0, 0.0)
    extents = Bbox(x0=80.0, y0=52.8, x1=576.0, y1=422.4)
    facecolor or fc = (1.0, 1.0, 1.0, 1.0)
    figure = Figure(640x480)
    fill = True
    gid = None
    hatch = None
    height = 1.0
    in_layout = True
    joinstyle = miter
    label =
    linestyle or ls = solid
    linewidth or lw = 0.0
    mouseover = False
    patch_transform = CompositeGenericTransform(     BboxTransformTo(   ...
    path = Path(array([[0., 0.],        [1., 0.],        [1.,...
    path_effects = []
    picker = None
    rasterized = False
    sketch_params = None
    snap = None
    tightbbox = Bbox(x0=80.0, y0=52.8, x1=576.0, y1=422.4)
    transform = CompositeGenericTransform(     CompositeGenericTra...
    transformed_clip_path_and_affine = (None, None)
    url = None
    verts = [[ 80.   52.8]  [576.   52.8]  [576.  422.4]  [ 80...
    visible = True
    width = 1.0
    window_extent = Bbox(x0=80.0, y0=52.8, x1=576.0, y1=422.4)
    x = 0.0
    xy = (0.0, 0.0)
    y = 0.0
    zorder = 1
Text setters
  agg_filter: a filter function, which takes a (m, n, 3) float array and a dpi value, and returns a (m, n, 3) array and two offsets from the bottom left corner of the image
  alpha: scalar or None
  animated: bool
  backgroundcolor: color
  bbox: dict with properties for `.patches.FancyBboxPatch`
  clip_box: `.Bbox`
  clip_on: bool
  clip_path: Patch or (Path, Transform) or None
  color or c: color
  figure: `.Figure`
  fontfamily or family: {FONTNAME, 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'}
  fontproperties or font or font_properties: `.font_manager.FontProperties` or `str` or `pathlib.Path`
  fontsize or size: float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
  fontstretch or stretch: {a numeric value in range 0-1000, 'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed', 'normal', 'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'}
  fontstyle or style: {'normal', 'italic', 'oblique'}
  fontvariant or variant: {'normal', 'small-caps'}
  fontweight or weight: {a numeric value in range 0-1000, 'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'}
  gid: str
  horizontalalignment or ha: {'left', 'center', 'right'}
  in_layout: bool
  label: object
  linespacing: float (multiple of font size)
  math_fontfamily: str
  mouseover: bool
  multialignment or ma: {'left', 'right', 'center'}
  parse_math: bool
  path_effects: `.AbstractPathEffect`
  picker: None or bool or float or callable
  position: (float, float)
  rasterized: bool
  rotation: float or {'vertical', 'horizontal'}
  rotation_mode: {None, 'default', 'anchor'}
  sketch_params: (scale: float, length: float, randomness: float)
  snap: bool or None
  text: object
  transform: `.Transform`
  transform_rotates_text: bool
  url: str
  usetex: bool or None
  verticalalignment or va: {'bottom', 'baseline', 'center', 'center_baseline', 'top'}
  visible: bool
  wrap: bool
  x: float
  y: float
  zorder: float
Text getters
    agg_filter = None
    alpha = None
    animated = False
    bbox_patch = None
    children = []
    clip_box = None
    clip_on = True
    clip_path = None
    color or c = black
    figure = Figure(640x480)
    fontfamily or family = ['sans-serif']
    fontname or name = DejaVu Sans
    fontproperties or font or font_properties = sans\-serif:style=normal:variant=normal:weight=nor...
    fontsize or size = 12.0
    fontstyle or style = normal
    fontvariant or variant = normal
    fontweight or weight = normal
    gid = None
    horizontalalignment or ha = center
    in_layout = True
    label =
    math_fontfamily = dejavusans
    mouseover = False
    parse_math = True
    path_effects = []
    picker = None
    position = (0.5, 1.0)
    rasterized = False
    rotation = 0.0
    rotation_mode = None
    sketch_params = None
    snap = None
    stretch = normal
    text = Hi mom
    tightbbox = Bbox(x0=295.5, y0=426.7333333333333, x1=360.5, y1=...
    transform = CompositeGenericTransform(     BboxTransformTo(   ...
    transform_rotates_text = False
    transformed_clip_path_and_affine = (None, None)
    unitless_position = (0.5, 1.0)
    url = None
    usetex = False
    verticalalignment or va = baseline
    visible = True
    window_extent = Bbox(x0=295.5, y0=426.7333333333333, x1=360.5, y1=...
    wrap = False
    zorder = 3


 HOW TO SIZE PLOTS

 
import matplotlib.pyplot as plt
import numpy as np

data = np.random.random(size=100)
fig, axes = plt.subplots(2, 2, figsize=(10, 5), squeeze = False)

axes[0,0].plot(data)
axes[0,0].set_title("\n".join(["this is a really long title"]*2))
axes[0,1].plot(data)
axes[1,1].plot(data)

fig.suptitle("\n".join(["a big long suptitle that runs into the title"]*2), y=0.98)


def make_space_above(axes, topmargin=1):
    """ increase figure size to make topmargin (in inches) space for 
        titles, without changing the axes sizes"""
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1-s.top)*h  + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)


make_space_above(axes, topmargin=1)    

plt.show()


'''
# def get_fig_shape(n_plots):
#     n_col = int(np.ceil(np.sqrt(n_plots)))
#     n_plots_sq = n_col**2
#     diff = n_plots_sq - n_plots
#     n_row = n_col
#     if diff >= n_col:
#         n_row -= 1
#     return n_col, n_row  # first is the bottom axis, second is the top axis


# def get_fig_size(n_col, n_row, ratio=1, base=5, scaling=0.85):
#     additional_space_a = [base * scaling**x for x in range(1, n_col+1)]
#     additional_space_b = [ratio * base * scaling**x for x in range(1, n_row+1)]
#     return (sum(additional_space_a), sum(additional_space_b))


# def plot(xdata: Union[np.ndarray, list, List[list]], 
#          ydata: Union[np.ndarray, list, List[list], None]=None, 
#          labels: Union[List[list], None]=None, 
         
#          xlabel: Union[str, list, None]=None,  
#          ylabel: Union[str, list, None]=None, 
#          title: Union[str, list, None]=None,

#          color: Union[str, list, None]='blue',

#          hline: Union[float, list, dict, None]=None,
#          vline: Union[float, list, dict, None]=None,

#          plot_type: str='line', 
#          fig_title: str=None,
#          fig_path: str=None,
#          **kwargs):
#     '''

#     nb
#     - if multiple h/v lines repeated make a list of lists
#     - for data_labels (key) only need in case of multiple lines
#     '''
    
#     args = locals()
#     args = {k:v for k, v in args.items() if not 'fig' in k}

#     nx = len(xdata) if isinstance(xdata, list) else 1
#     ny = len(ydata) if isinstance(ydata, list) else 1
#     n_plots = max(nx, ny)

#     n_col, n_row = get_fig_shape(n_plots)
#     figsize = get_fig_size(n_col, n_row)
#     fig, axs = plt.subplots(ncols=n_col, nrows=n_row, figsize=figsize)  # a columns, b rows
    
#     if not isinstance(axs, Iterable):
#         axs = [axs]
    
#     if n_col > 1 and n_row > 1:
#         axs = axs.flatten() # creates an iterator
    

#     # each element of this list
#     # is a dictionary of the args for that plot
#     # and accounts for 
#     # 1) argument not provided as list (applies to all)
#     # 2) provided as list but only 1 element (list of list hlines case)
#     # 3) provided for all plots
#     # Fail case: will fail if len(v) not equal 1 or n_plots
#     # Fixed now copies to all future
#     args = [{k:v if not isinstance(v, list) else v[min(len(v)-1, i)] for k, v in args.items()} for i in range(n_plots)]

#     for i, (ax, arg) in enumerate(zip(axs, args)):
#         plot_kwargs = {k:v for k,v in arg.items() if k in plot_arguments}
        
#         if plot_type == 'line':

#             ax.plot(
#                 arg['xdata'],
#                 arg['ydata'],
#                 **plot_kwargs
#             )

#         if plot_type == 'hist':
#             '''
#             bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None, 
#             histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, 
#             color=None, label=None, stacked=False, *, data=None, **kwargs
#             '''
#             ax.hist(
#                 arg.get('x'),
#                 **kwargs
#             )

#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()

#         for fn, lines, lims in zip([ax.hlines, ax.vlines], ['hlines', 'vlines'], [xlim, ylim]):
#             if not arg.get(lines) is None:
#                 if isinstance(arg[lines], dict):
#                     kwargs = {k:v for k,v in arg[lines].items() if not k == 'lines'}
#                     lines = arg[lines]['lines']
#                 else:
#                     kwargs = {}
#                     lines = arg[lines]

#                 fn(lines, *lims, **kwargs)

#         ax.set_xlabel(arg.get('xlabel'))
#         ax.set_ylabel(arg.get('ylabel'))
#         ax.set_title(arg.get('title'))

#         ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 3), useMathText=True, useOffset=True)
#         ax.tick_params(axis='both', pad=3.)  # pad default is 4.

#     [ax.remove() for ax in axs[n_plots:]]

#     fig.suptitle(fig_title)
#     fig.tight_layout()
#     if fig_path is not None: 
#         fig.savefig(fig_path)

#     return fig