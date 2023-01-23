
systems = dict(
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



