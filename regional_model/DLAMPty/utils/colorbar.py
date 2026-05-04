import matplotlib.colors as mcolors

# wsp_lev = [0,4,6,8,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,36,38,40,43,46,49,52,55,58,61,64,67,70,73,76,79,82,85]
# wsp_color = ['#ffffff','#80ffff','#6fedf1','#5fdde4','#50cdd5','#40bbc7','#2facba','#1f9bac','#108c9f','#007a92',
#              '#00b432','#33c341','#67d251','#99e060','#cbf06f','#ffff80','#ffdd52','#ffdc52','#ffa63e','#ff6d29','#ff3713','#ff0000','#d70000','#af0000','#870000','#5f0000',
#              '#aa00ff','#b722fe','#c446ff','#d46aff','#e38dff','#f1b1ff','#ffd3ff',
#              '#ffc6ea','#ffb6d5','#ffa6c1','#ff97ac','#ff8798','#fe7884','#ff696e','#ff595a','#e74954','#cc3a4c','#b22846','#9a1941']


def clist_WS():
    clist = [
        (0,       "#ffffff"),
        (3 / 40,  "#80ffff"),
        (6 / 40,  "#6fedf1"),
        (8 / 40,  "#5fdde4"),
        (10 / 40, "#50cdd5"),
        (12 / 40, "#40bbc7"),
        (13 / 40, "#2facba"),
        (14 / 40, "#1f9bac"),
        (15 / 40, "#108c9f"),
        (16 / 40, "#007a92"),
        (17 / 40, "#00b432"),
        (18 / 40, "#33c341"),
        (19 / 40, "#67d251"),
        (20 / 40, "#99e060"),
        (21 / 40, "#cbf06f"),
        (22 / 40, "#ffff80"),
        (23 / 40, "#ffdd52"),
        (24 / 40, "#ffdc52"),
        (25 / 40, "#ffa63e"),
        (26 / 40, "#ff6d29"),
        (27 / 40, "#ff3713"),
        (28 / 40, "#ff0000"),
        (29 / 40, "#d70000"),
        (30 / 40, "#af0000"),
        (31 / 40, "#870000"),
        (32 / 40, "#5f0000"),
        (34 / 40, "#aa00ff"),
        (36 / 40, "#b722fe"),
        (38 / 40, "#c446ff"),
        (40 / 40, "#d46aff"),
    ]
    return clist


def clist_temp():
    clist = [
        (0,       "#a8acdf"),
        (1 / 40,  "#9092d4"),
        (2 / 40,  "#777acc"),
        (3 / 40,  "#5f63c3"),
        (4 / 40,  "#4949b6"),
        (5 / 40,  "#4655c3"),
        (6 / 40,  "#435aca"),
        (7 / 40,  "#3b6ddf"),
        (8 / 40,  "#3979ef"),
        (9 / 40,  "#3386f5"),
        (10 / 40, "#2d99fe"),
        (11 / 40, "#22affe"),
        (12 / 40, "#1bc2ff"),
        (13 / 40, "#0ee6fe"),
        (14 / 40, "#07fbff"),
        (15 / 40, "#6ee699"),
        (16 / 40, "#65e08d"),
        (17 / 40, "#4fd06f"),
        (18 / 40, "#45c65f"),
        (19 / 40, "#34bd4b"),
        (20 / 40, "#28b338"),
        (21 / 40, "#16a71f"),
        (22 / 40, "#16a111"),
        (23 / 40, "#43b121"),
        (24 / 40, "#66c034"),
        (25 / 40, "#78c63c"),
        (26 / 40, "#9ad54d"),
        (27 / 40, "#c5e763"),
        (28 / 40, "#e1f26f"),
        (29 / 40, "#fef87b"),
        (30 / 40, "#fdeb76"),
        (31 / 40, "#fad66a"),
        (32 / 40, "#f9c662"),
        (33 / 40, "#f8b558"),
        (34 / 40, "#f6a24e"),
        (35 / 40, "#ef9043"),
        (36 / 40, "#e4692c"),
        (37 / 40, "#e15f27"),
        (38 / 40, "#cc3513"),
        (39 / 40, "#c8250a"),
        (40 / 40, "#c8250a"),
    ]
    return clist

prec_colorlist = [
    "#ffffff",  # 0.01 - 0.10 inches
    "#c9c9c9",  # 0.10 - 0.25 inches
    "#9dfeff",
    "#01d2fd",  # 0.25 - 0.50 inches
    "#00a5fe",  # 0.50 - 0.75 inches
    "#0177fd",  # 0.75 - 1.00 inches
    "#27a31b",  # 1.00 - 1.50 inches
    "#00fa2f",  # 1.50 - 2.00 inches
    "#fffe33",  # 2.00 - 2.50 inches
    "#ffd328",  # 2.50 - 3.00 inches
    "#ffa71f",  # 3.00 - 4.00 inches
    "#ff2b06",
    "#da2304",  # 4.00 - 5.00 inches
    "#aa1801",  # 5.00 - 6.00 inches
    "#ab1fa2",  # 6.00 - 8.00 inches
    "#db2dd2",  # 8.00 - 10.00 inches
    "#ff38fb",  # 10.00+
    "#ffd5fd",
]

prec_levels = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20]

discrete_prec_cmap = mcolors.ListedColormap(prec_colorlist)
prec_norm = mcolors.BoundaryNorm(prec_levels, ncolors=discrete_prec_cmap.N, clip=True)

def calc_mids(levels):
    levels = [levels[i]-levels[0] for i in range(len(levels))]
    boundarys = [levels[i]/max(levels) for i in range(len(levels)-2)]
    middles = [boundarys[1]/2.]
    for i in range(1, len(boundarys)):
        middles.append(boundarys[i]*2. - middles[i-1])
    middles = [middles[i]-middles[0] for i in range(len(middles))]
    middles.append(1)
    return middles

def clist_prec():
    return list(zip(calc_mids(prec_levels), prec_colorlist))

vort_colorlist = [
    "#1464d3",  # < -15
    "#2a81f2",  # < -10
    "#50a6f1",  # <  -6
    "#97d2fa",  # <  -4
    "#e1ffff",  # <  -2
    "#ffffff",  # <   2
    "#fff8ab",  # <   4
    "#fdc13c",  # <   6
    "#ff6100",  # <  10
    "#e11400",  # <  15
    "#a40000",  # above 15
]

vort_levels = [-80, -60, -40, -24, -16,  -8,   8,  16,  24,  40,  60,  80]

vort_cmap = mcolors.ListedColormap(vort_colorlist)
vort_norm = mcolors.BoundaryNorm(vort_levels, ncolors=vort_cmap.N, clip=True)

def clist_vort():
    vort_mids = [
        -16 /32+0.5,
        -13 /32+0.5,
         -8 /32+0.5,
         -5 /32+0.5,
         -3 /32+0.5,
          0 /32+0.5,
          3 /32+0.5,
          5 /32+0.5,
          8 /32+0.5,
         13 /32+0.5,
         16 /32+0.5,
    ]
    return list(zip(vort_mids, vort_colorlist))

def make_cmap(clist_name):
    clist = globals()[clist_name]()
    name = clist_name.split("_")[1]
    cmap = mcolors.LinearSegmentedColormap.from_list(f"my_{name}", clist)
    return cmap
