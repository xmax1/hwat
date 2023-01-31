
systems = dict(
    o2test = dict(
        nametag = 'O2_Triplet',
        precision = '+',
        charge = 0,
        spin = -2,
        a = [[0,0,0],[0,0,1.2075]],
        a_z = [8.,8.],
        n_b = 64,
        n_pre_step = 20,
        n_step = 40,
        
        info = 'The ground state of oxygen with two unpaired electrons, i.e. spin multiplicity of 3',
    ),

    Be = dict(
        nametag = 'Beryllium',
        precision = '+',
        charge = 1,
        spin = -1,
        a = [[0.0, 0.0, 0.0],],
        a_z = [4.,],
        info = 'I have wanted to simulate this system since I learned about beryllium in chemistry at school :)',
    ),

    Ne = dict(
        nametag = 'Neon',
        precision = '+',
        charge = 0,
        spin = 0,
        a = [[0.0, 0.0, 0.0],],
        a_z = [10.,],
        info = 'Baseline',
    ),

    O2_neutral_triplet = dict(
        nametag = 'O2_Triplet',
        precision = '+',
        charge = 0,
        spin = -2,
        a = [[0,0,0],[0,0,1.2075]],
        a_z = [8.,8.],
        info = 'The ground state of oxygen with two unpaired electrons, i.e. spin multiplicity of 3',
    ),

    O2_neutral_singlet = dict(
        nametag = 'O2_Singlet',
        precision = '+',
        charge = 0,
        spin = 0,
        a = [[0,0,0],[0,0,1.2255]],
        a_z = [8,8],
        info = 'The first excited state of oxygen with no overall spin, i.e. spin multiplicity of 1',
    ),

    O2_oxidized_doublet = dict(
        nametag = 'O2_Oxidized_Doublet',
        precision = '+',
        charge = 1,
        spin = -1,
        a = [[0,0,0],[0,0,1.1164]],
        a_z = [8,8],
        info = 'A singly oxidized molecular oxygen with one unpaired electron, i.e. spin multiplicity of 2',
    ),
)




"""



This file contains various system configurations, e.g., atomic, 
diatomic systems, cyclobutadiene, ...
import numpy as np

from .element import ELEMENT_BY_ATOMIC_NUM, ELEMENT_BY_SYMBOL
from .system import Atom, Molecule


def atomic(symbol: str, *args, **kwargs):
    return Molecule([
        Atom(symbol, (0.0, 0.0, 0.0))
    ], **kwargs)


def diatomic(symbol1: str, symbol2: str, R: float, units='bohr', *args, **kwargs):
    return Molecule([
        Atom(symbol1, (-R/2, 0.0, 0.0), units=units),
        Atom(symbol2, (R/2, 0.0, 0.0), units=units),
    ])


def H2(R: float, *args, **kwargs):
    return diatomic('H', 'H', R, *args, **kwargs)


def h4_plane(theta: float, R: float, *args, **kwargs):
    y = np.sin(np.radians(theta/2)) * R
    x = np.cos(np.radians(theta/2)) * R
    return Molecule([
        Atom('H', (x, y, 0.0)),
        Atom('H', (x, -y, 0.0)),
        Atom('H', (-x, y, 0.0)),
        Atom('H', (-x, -y, 0.0))
    ])


def h_chain(n: int, R: float, *args, **kwargs):
    span = (n-1)*R
    center = span/2
    return Molecule([
        Atom('H', (i*R - center, 0.0, 0.0))
        for i in range(n)
    ])


def by_positions(symbols, positions, units='bohr', spins=None, *args, **kwargs):
    assert len(symbols) == len(positions)
    positions = np.array(positions)
    return Molecule([
        Atom(sym, coords)
        for sym, coords in zip(symbols, positions)
    ], spins=spins)


def h4plus(positions: np.ndarray, *args, **kwargs):
    return by_positions(['H']*len(positions), positions, spins=(2, 1))


def cyclobutadiene(state: str):
    # https://github.com/deepmind/ferminet/blob/jax/ferminet/configs/organic.py
    if state == 'ground':
        return Molecule([
            Atom('C', (0.0000000e+00, 0.0000000e+00, 0.0000000e+00)),
            Atom('C', (2.9555318e+00, 0.0000000e+00, 0.0000000e+00)),
            Atom('C', (2.9555318e+00, 2.5586891e+00, 0.0000000e+00)),
            Atom('C', (0.0000000e+00, 2.5586891e+00, 0.0000000e+00)),
            Atom('H', (-1.4402903e+00, -1.4433100e+00, 1.7675451e-16)),
            Atom('H', (4.3958220e+00, -1.4433100e+00, -1.7675451e-16)),
            Atom('H', (4.3958220e+00, 4.0019994e+00, 1.7675451e-16)),
            Atom('H', (-1.4402903e+00, 4.0019994e+00, -1.7675451e-16)),
        ])
    elif state == 'transition':
        return Molecule([
            Atom('C', (0.0000000e+00, 0.0000000e+00, 0.0000000e+00)),
            Atom('C', (2.7419927e+00, 0.0000000e+00, 0.0000000e+00)),
            Atom('C', (2.7419927e+00, 2.7419927e+00, 0.0000000e+00)),
            Atom('C', (0.0000000e+00, 2.7419927e+00, 0.0000000e+00)),
            Atom('H', (-1.4404647e+00, -1.4404647e+00, 1.7640606e-16)),
            Atom('H', (4.1824574e+00, -1.4404647e+00, -1.7640606e-16)),
            Atom('H', (4.1824574e+00, 4.1824574e+00, 1.7640606e-16)),
            Atom('H', (-1.4404647e+00, 4.1824574e+00, -1.7640606e-16))
        ])


def bicyclobutane(state: str):
    # https://pubs.acs.org/doi/abs/10.1021/jp065721k
    # Supplementary
    # https://github.com/deepmind/ferminet/blob/jax/ferminet/configs/organic.py
    if state == 'bicbut':
        return Molecule([
            Atom('C', (1.0487346562, 0.5208579773, 0.2375867187), units='angstrom'),
            Atom('C', (0.2497284256, -0.7666691493, 0.0936474818), units='angstrom'),
            Atom('C', (-0.1817326465, 0.4922777820, -0.6579637266), units='angstrom'),
            Atom('C', (-1.1430708301, -0.1901383337, 0.3048494250), units='angstrom'),
            Atom('H', (2.0107137141, 0.5520589541, -0.2623459977), units='angstrom'),
            Atom('H', (1.0071921280, 1.0672669240, 1.1766131856), units='angstrom'),
            Atom('H', (0.5438033167, -1.7129829738, -0.3260782874), units='angstrom'),
            Atom('H', (-0.2580605320, 0.6268443026, -1.7229636111), units='angstrom'),
            Atom('H', (-1.3778676954, 0.2935640723, 1.2498189977), units='angstrom'),
            Atom('H', (-1.9664163102, -0.7380906148, -0.1402911727), units='angstrom')
        ])
    elif state == 'con_TS':
        return Molecule([
            Atom('C', (1.0422528085, 0.5189448459, 0.2893513723), units='angstrom'),
            Atom('C', (0.6334392052, -0.8563584473, -0.1382423606), units='angstrom'),
            Atom('C', (-0.2492035181, 0.3134656784, -0.5658962512), units='angstrom'),
            Atom('C', (-1.3903646889, 0.0535204487, 0.2987506023), units='angstrom'),
            Atom('H', (1.8587636947, 0.9382817031, -0.2871146890), units='angstrom'),
            Atom('H', (0.9494853889, 0.8960565051, 1.3038563129), units='angstrom'),
            Atom('H', (0.3506375894, -1.7147937260, 0.4585707483), units='angstrom'),
            Atom('H', (-0.3391417369, 0.6603641863, -1.5850373819), units='angstrom'),
            Atom('H', (-1.2605467656, 0.0656225945, 1.3701508857), units='angstrom'),
            Atom('H', (-2.3153892612, -0.3457478660, -0.0991685880), units='angstrom'),
        ])
    elif state == 'dis_TS':
        return Molecule([
            Atom('C', (1.5864390444, -0.1568990400, -0.1998155990), units='angstrom'),
            Atom('C', (-0.8207390911, 0.8031532550, -0.2771554962), units='angstrom'),
            Atom('C', (0.2514913592, 0.0515423448, 0.4758741643), units='angstrom'),
            Atom('C', (-1.0037104567, -0.6789877402, -0.0965401189), units='angstrom'),
            Atom('H', (2.4861305372, 0.1949133826, 0.2874101433), units='angstrom'),
            Atom('H', (1.6111805503, -0.2769458302, -1.2753251100), units='angstrom'),
            Atom('H', (-1.4350764228, 1.6366792379, 0.0289087336), units='angstrom'),
            Atom('H', (0.2833919284, 0.1769734467, 1.5525271253), units='angstrom'),
            Atom('H', (-1.7484283536, -1.0231589431, 0.6120702030), units='angstrom'),
            Atom('H', (-0.8524391649, -1.3241689195, -0.9544331346), units='angstrom')
        ])
    elif state == 'g-but':
        return Molecule([
            Atom('C', (1.4852019019, 0.4107781008, 0.5915178362), units='angstrom'),
            Atom('C', (0.7841417614, -0.4218449588, -0.2276848579), units='angstrom'),
            Atom('C', (-0.6577970182, -0.2577617373, -0.6080850660), units='angstrom'),
            Atom('C', (-1.6247236649, 0.2933006709, 0.1775352473), units='angstrom'),
            Atom('H', (1.0376813593, 1.2956518484, 1.0267024109), units='angstrom'),
            Atom('H', (2.5232360753, 0.2129135014, 0.8248568552), units='angstrom'),
            Atom('H', (1.2972328960, -1.2700686671, -0.6686116041), units='angstrom'),
            Atom('H', (-0.9356614935, -0.6338686329, -1.5871170536), units='angstrom'),
            Atom('H', (-1.4152018269, 0.6472889925, 1.1792563311), units='angstrom'),
            Atom('H', (-2.6423222755, 0.3847635835, -0.1791755263), units='angstrom')
        ])
    elif state == 'gt_TS':
        return Molecule([
            Atom('C', (1.7836595975, 0.4683155866, -0.4860478101), units='angstrom'),
            Atom('C', (0.7828892933, -0.4014025715, -0.1873880949), units='angstrom'),
            Atom('C', (-0.6557274850, -0.2156646805, -0.6243545354), units='angstrom'),
            Atom('C', (-1.6396999531, 0.2526943506, 0.1877948644), units='angstrom'),
            Atom('H', (1.6003117673, 1.3693309737, -1.0595471944), units='angstrom'),
            Atom('H', (2.7986234673, 0.2854595500, -0.1564989895), units='angstrom'),
            Atom('H', (1.0128486304, -1.2934621995, 0.3872559845), units='angstrom'),
            Atom('H', (-0.9003245968, -0.4891235826, - 1.6462438855), units='angstrom'),
            Atom('H', (-1.4414954784, 0.5345813494, 1.2152198579), units='angstrom'),
            Atom('H', (-2.6556262424, 0.3594422237, -0.1709361970), units='angstrom')
        ])
    elif state == 't-but':
        return Molecule([
            Atom('C', (0.6109149108, 1.7798412991, -0.0000000370), units='angstrom'),
            Atom('C', (0.6162339625, 0.4163908910, -0.0000000070), units='angstrom'),
            Atom('C', (-0.6162376752, -0.4163867945, -0.0000000601), units='angstrom'),
            Atom('C', (-0.6109129465, -1.7798435851, 0.0000000007), units='angstrom'),
            Atom('H', (1.5340442204, 2.3439205382, 0.0000000490), units='angstrom'),
            Atom('H', (-0.3156117962, 2.3419017314, 0.0000000338), units='angstrom'),
            Atom('H', (1.5642720455, -0.1114324578, -0.0000000088), units='angstrom'),
            Atom('H', (-1.5642719469, 0.1114307897, -0.0000000331), units='angstrom'),
            Atom('H', (-1.5340441021, -2.3439203971, 0.0000000714), units='angstrom'),
            Atom('H', (0.3156133277, -2.3419020150, -0.0000000088), units='angstrom')
        ])
    else:
        raise ValueError()

# https://github.com/deepmind/ferminet/blob/jax/ferminet/utils/elements.py
# This file is largely taken from Spencer et al., 2020.
class Element:

    Simple data class to manage basic information about elements.


    def __init__(self, symbol, atomic_number, period, spin=0) -> None:
        self.symbol = symbol
        self.atomic_number = atomic_number
        self.period = period
        self.spin = spin


# Static array of all relevant elements.
_ELEMENTS = (
    Element(symbol='H', atomic_number=1, period=1),
    Element(symbol='He', atomic_number=2, period=1),
    Element(symbol='Li', atomic_number=3, period=2),
    Element(symbol='Be', atomic_number=4, period=2),
    Element(symbol='B', atomic_number=5, period=2),
    Element(symbol='C', atomic_number=6, period=2),
    Element(symbol='N', atomic_number=7, period=2),
    Element(symbol='O', atomic_number=8, period=2),
    Element(symbol='F', atomic_number=9, period=2),
    Element(symbol='Ne', atomic_number=10, period=2),
    Element(symbol='Na', atomic_number=11, period=3),
    Element(symbol='Mg', atomic_number=12, period=3),
    Element(symbol='Al', atomic_number=13, period=3),
    Element(symbol='Si', atomic_number=14, period=3),
    Element(symbol='P', atomic_number=15, period=3),
    Element(symbol='S', atomic_number=16, period=3),
    Element(symbol='Cl', atomic_number=17, period=3),
    Element(symbol='Ar', atomic_number=18, period=3),
    Element(symbol='K', atomic_number=19, period=4),
    Element(symbol='Ca', atomic_number=20, period=4),
    Element(symbol='Sc', atomic_number=21, period=4, spin=1),
    Element(symbol='Ti', atomic_number=22, period=4, spin=2),
    Element(symbol='V', atomic_number=23, period=4, spin=3),
    Element(symbol='Cr', atomic_number=24, period=4, spin=6),
    Element(symbol='Mn', atomic_number=25, period=4, spin=5),
    Element(symbol='Fe', atomic_number=26, period=4, spin=4),
    Element(symbol='Co', atomic_number=27, period=4, spin=3),
    Element(symbol='Ni', atomic_number=28, period=4, spin=2),
    Element(symbol='Cu', atomic_number=29, period=4, spin=1),
    Element(symbol='Zn', atomic_number=30, period=4, spin=0),
    Element(symbol='Ga', atomic_number=31, period=4),
    Element(symbol='Ge', atomic_number=32, period=4),
    Element(symbol='As', atomic_number=33, period=4),
    Element(symbol='Se', atomic_number=34, period=4),
    Element(symbol='Br', atomic_number=35, period=4),
    Element(symbol='Kr', atomic_number=36, period=4),
    Element(symbol='Rb', atomic_number=37, period=5),
    Element(symbol='Sr', atomic_number=38, period=5),
    Element(symbol='Y', atomic_number=39, period=5, spin=1),
    Element(symbol='Zr', atomic_number=40, period=5, spin=2),
    Element(symbol='Nb', atomic_number=41, period=5, spin=5),
    Element(symbol='Mo', atomic_number=42, period=5, spin=6),
    Element(symbol='Tc', atomic_number=43, period=5, spin=5),
    Element(symbol='Ru', atomic_number=44, period=5, spin=4),
    Element(symbol='Rh', atomic_number=45, period=5, spin=3),
    Element(symbol='Pd', atomic_number=46, period=5, spin=0),
    Element(symbol='Ag', atomic_number=47, period=5, spin=1),
    Element(symbol='Cd', atomic_number=48, period=5, spin=0),
    Element(symbol='In', atomic_number=49, period=5),
    Element(symbol='Sn', atomic_number=50, period=5),
    Element(symbol='Sb', atomic_number=51, period=5),
    Element(symbol='Te', atomic_number=52, period=5),
    Element(symbol='I', atomic_number=53, period=5),
    Element(symbol='Xe', atomic_number=54, period=5),
    Element(symbol='Cs', atomic_number=55, period=6),
    Element(symbol='Ba', atomic_number=56, period=6),
    Element(symbol='La', atomic_number=57, period=6),
    Element(symbol='Ce', atomic_number=58, period=6),
    Element(symbol='Pr', atomic_number=59, period=6),
    Element(symbol='Nd', atomic_number=60, period=6),
    Element(symbol='Pm', atomic_number=61, period=6),
    Element(symbol='Sm', atomic_number=62, period=6),
    Element(symbol='Eu', atomic_number=63, period=6),
    Element(symbol='Gd', atomic_number=64, period=6),
    Element(symbol='Tb', atomic_number=65, period=6),
    Element(symbol='Dy', atomic_number=66, period=6),
    Element(symbol='Ho', atomic_number=67, period=6),
    Element(symbol='Er', atomic_number=68, period=6),
    Element(symbol='Tm', atomic_number=69, period=6),
    Element(symbol='Yb', atomic_number=70, period=6),
    Element(symbol='Lu', atomic_number=71, period=6),
    Element(symbol='Hf', atomic_number=72, period=6),
    Element(symbol='Ta', atomic_number=73, period=6),
    Element(symbol='W', atomic_number=74, period=6),
    Element(symbol='Re', atomic_number=75, period=6),
    Element(symbol='Os', atomic_number=76, period=6),
    Element(symbol='Ir', atomic_number=77, period=6),
    Element(symbol='Pt', atomic_number=78, period=6),
    Element(symbol='Au', atomic_number=79, period=6),
    Element(symbol='Hg', atomic_number=80, period=6),
    Element(symbol='Tl', atomic_number=81, period=6),
    Element(symbol='Pb', atomic_number=82, period=6),
    Element(symbol='Bi', atomic_number=83, period=6),
    Element(symbol='Po', atomic_number=84, period=6),
    Element(symbol='At', atomic_number=85, period=6),
    Element(symbol='Rn', atomic_number=86, period=6),
    Element(symbol='Fr', atomic_number=87, period=7),
    Element(symbol='Ra', atomic_number=88, period=7),
    Element(symbol='Ac', atomic_number=89, period=7),
    Element(symbol='Th', atomic_number=90, period=7),
    Element(symbol='Pa', atomic_number=91, period=7),
    Element(symbol='U', atomic_number=92, period=7),
    Element(symbol='Np', atomic_number=93, period=7),
    Element(symbol='Pu', atomic_number=94, period=7),
    Element(symbol='Am', atomic_number=95, period=7),
    Element(symbol='Cm', atomic_number=96, period=7),
    Element(symbol='Bk', atomic_number=97, period=7),
    Element(symbol='Cf', atomic_number=98, period=7),
    Element(symbol='Es', atomic_number=99, period=7),
    Element(symbol='Fm', atomic_number=100, period=7),
    Element(symbol='Md', atomic_number=101, period=7),
    Element(symbol='No', atomic_number=102, period=7),
    Element(symbol='Lr', atomic_number=103, period=7),
    Element(symbol='Rf', atomic_number=104, period=7),
    Element(symbol='Db', atomic_number=105, period=7),
    Element(symbol='Sg', atomic_number=106, period=7),
    Element(symbol='Bh', atomic_number=107, period=7),
    Element(symbol='Hs', atomic_number=108, period=7),
    Element(symbol='Mt', atomic_number=109, period=7),
    Element(symbol='Ds', atomic_number=110, period=7),
    Element(symbol='Rg', atomic_number=111, period=7),
    Element(symbol='Cn', atomic_number=112, period=7),
    Element(symbol='Nh', atomic_number=113, period=7),
    Element(symbol='Fl', atomic_number=114, period=7),
    Element(symbol='Mc', atomic_number=115, period=7),
    Element(symbol='Lv', atomic_number=116, period=7),
    Element(symbol='Ts', atomic_number=117, period=7),
    Element(symbol='Og', atomic_number=118, period=7),
)

ELEMENT_BY_SYMBOL = {e.symbol: e for e in _ELEMENTS}
ELEMENT_BY_ATOMIC_NUM = {e.atomic_number: e for e in _ELEMENTS}
        
"""


