# HeckeAlgebras.jl

A Julia package for computing in (Iwahori–)Hecke algebras for Coxeter groups (draft, under development).

The current code is due to M. Albert (2022) and is based on the Magma package [IHecke](https://github.com/joelgibson/IHecke) by J. Gibson.

## Installation

The computation in Coxeter groups relies on the package [CoxeterGroups.jl](https://github.com/ulthiel/CoxeterGroups.jl), which has to be installed as well:

```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/ulthiel/CoxeterGroups.jl")
julia> Pkg.add(url="https://github.com/ulthiel/HeckeAlgebras.jl")
```

## Usage


```julia
julia> using HeckeAlgebras

julia> W, (a, b, c) = CoxeterGroup([1 3 2; 3 1 4; 2 4 1], ["a", "b", "c"]);

julia> HAlg = HeckeAlgebra(W);

# A Hecke algebra created as above does not have a distinguished basis yet. 
# Typically, one will work with the standard basis and the Kazhdan-Lusztig basis.
# This is created as follows.
julia> H = StandardBasis(HAlg);
julia> K = KazhdanLusztigBasis(HAlg);
julia> A, v = LaurentPolynomialRing(ZZ, "v");

# Now, we can work with the basis elements. They are accessed via bracket notation
# encoding the elements as word in the Coxeter group, e.g.
julia> H[1]
(1)*H[1]

# Some arithmetic
julia> H[3]*H[1]
(1)*H[3, 1]

julia> H[3]*K[2]
(v)*H[3] + (1)*H[3, 2]

julia> H[3]*(H[1] + v*H[[1, 2, 3]])
(v)*H[3, 1, 2, 3] + (1)*H[3, 1]

julia> 2*H[3]*K[2]^3 + (v^2 + v^-1)*K[1]
(2*v^3 + 4*v + 2*v^-1)*H[3] + (v^2 + v^-1)*H[1] + (v^3 + 1)*H[id] + (2*v^2 + 4 + 2*v^-2)*H[3, 2]

# The bar involution:
julia> Bar(K[1])
(1)*K[1]

julia> Bar(H[3])
(1)*H[3] + (v - v^-1)*H[id]

# Writing a Kazhdan-Lusztig basis element in the standard basis:
julia> H:K[2]
(v)*H[id] + (1)*H[2]
```

## Todo

