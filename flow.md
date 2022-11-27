
# Train

```mermaid
graph LR
    sys[System <br> Coordinates] --> f{Features}
    sys --> pe_f
    
    subgraph Features
        f{Features} --> ss[Single <br> Electron]
        f{Features} --> ps[Paiwise <br> Electron]
    end
    An{Ansatz}
    ss --> An
    ps --> An

    An --> sng
    An --> psi
    subgraph Metropolis <br> Hastings
        sng[sign_psi]
        psi[log_psi]
    end

    psi --> grad
    psi --> ke_f

    subgraph Energy 
        ke_f{Laplacian} --> ke[Kinetic]
        pe_f{Potential} --> |Get money| pe((Potenial))
        ke --> e[Energy]
        pe --> e
    end

    
    e --> update
    subgraph Compute <br> Grad
        grad{Grad}
    end
    grad --> update{Update}
```


# Model

```mermaid
graph LR
    x((walkers))
    psi[log_psi]
    sgn[sgn_psi]
    _psi['psi']

    x --> r((e-e <br> Distance))
    x --> d((e-e <br> Displacement))
    x --> rn((e-n <br> Distance))
    x --> dn((e-n <br> Displacement))
    x --> pos((e <br> Position))
    x --> posn((n <br> Position))

    x --> bf
    
    pos --> ind{Single Electron Features}
    posn --> ind
    d --> ind
    dn --> ind
    r --> dep

    ind --> bf[Backflow]
    subgraph Permutation Eqv. Layer<br><br><br><br><br>
        subgraph Single Electron Stream
            ind
        end
        subgraph Pairwise Electron Stream
            dep{Pairwise Electron Features}
        end
    end

    dep --> exp
    dep --> jas
    bf --> det
    subgraph Determinant
        subgraph Orbital
            exp[Exponent] --> det
            det[Determinant]
        end
    end
    subgraph Jastrow
        jas[Jastow]
    end
    jas --> _psi
    det --> _psi
    _psi --> psi
    _psi --> sgn
```