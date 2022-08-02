# By M. Albert (2022)
# Meine Version von Joel Gibsons Implementierung von Coxeter Gruppen.

module Coxeter_groups_MA
export CoxGrp, CoxElt, lexword, rank, gens, coxeter_group_mat, CoxEltMat, right_descent_set, left_descent_set, parent, mult_gen, to_cartan_matrix

import Nemo:
    CalciumField, diagonal_matrix, nrows, base_ring, identity_matrix, ca
import AbstractAlgebra:
    Generic.MatSpaceElem

"Abstract supertype for Coxeter groups"
abstract type CoxGrp end

"Abstract supertype for Coxeter group elements"
abstract type CoxElt end

"An iterator over some reduced expression for x."
Base.iterate(x::CoxElt) = iterate(x, x)

function Base.iterate(::CoxElt, x::CoxElt)
    desc = left_descent_set(x)
    if length(desc) == 0
        return nothing
    end
    s = first(desc)::Int64
    return (s, mult_gen(s, x))
end

"The length of a Coxeter group element is the length of any reduced expression in the generators."
Base.length(x::CoxElt) = count((_) -> true, x)

"Returns `true` if `x` is the identity element in the group."
Base.isone(x::CoxElt) = length(right_descent_set(x)) == 0

"The product ``xy`` of Coxeter group elements."
Base.:(*)(x::CoxElt, y::CoxElt) = foldl(mult_gen, y; init=x)

"The inverse ``x^-1`` of a Coxeter group element."
Base.inv(x::CoxElt) = foldl((y, s) -> mult_gen(s, y), x; init=one(parent(x)))

"The lexicographically least reduced expression for `x`"
function lexword(x::CoxElt)
    word = Array{Int, 1}()
    while true
        desc = left_descent_set(x)
        if length(desc) == 0
            return word
        end
        s = first(desc)
        push!(word, s)
        x = mult_gen(s, x)
    end
end

"The power ``x^n`` of a Coxeter group element."
function Base.:(^)(x::CoxElt, n::Integer)
    # Deal with negative powers
    if n < 0
        x = inv(x)
        n = -n
    end

    # Exponentiate by squaring
    acc = one(parent(x))
    pow2 = x
    while n > 0
        if n % 2 == 1
            acc *= pow2
        end
        pow2 *= pow2
        n ÷= 2
    end

    return acc
end

"""
Takes an integral matrix, which can be a Coxeter matrix or a crystallographic Cartan matrix, and
a compatible Cartan matrix.
"""
function to_cartan_matrix(mat)
    # The matrix must be square.
    m, n = size(mat)
    if m != n
        error("The matrix passed to to_cartan_matrix must be square.")
    end

    # Assume that mat is a Cartan matrix if the diagonal is all 2.
    if all(mat[s, s] == 2 for s=1:n)
        # Check the other Cartan matrix conditions
        if !all(mat[s, t] <= 0 for s=1:n, t=1:n if s != t)
            error("A Cartan matrix must have off-diagonal entries <= 0.")
        end
        if any(mat[s, t] != 0 for s=1:n, t=1:n if mat[t, s] == 0)
            error("A Cartan matrix must satisfy a_st = 0 iff a_ts == 0")
        end

        # If the Cartan matrix is integral, convert to a matrix over ZZ.
        if all(isinteger(mat[s, t]) for s=1:n, t=1:n)
            return matrix(ZZ, Matrix(mat))
        end

        return mat
    end

    # Assume that mat is a Coxeter matrix if the diagonal is all 1.
    if all(mat[s, s] == 1 for s=1:n)
        # Check the other Coxeter matrix conditions
        if !all(isinteger(mat[s, t]) && (mat[s, t] == 0 || mat[s, t] >= 2) for s=1:n, t=1:n if s != t)
            error("A Coxeter matrix must have off-diagonal entries in {0, 2, 3, 4, ...}")
        end
        if any(mat[s, t] != mat[t, s] for s=1:n, t=1:n)
            error("A Coxeter matrix must be symmetric")
        end

        C = CalciumField()
        A = diagonal_matrix(C(2), n, n)
        for s=1:n
            for t=1:n
                if mat[s, t] != 0
                    A[s, t] = -2 * cos(C(pi)//mat[s, t])
                else
                    A[s, t] = -2
                end
            end
        end
        return A
    end
    error("The matrix passed to to_cartan_matrix was neither a Cartan matrix nor a Coxeter matrix.")
end



"""
A Coxeter group represented by matrices of type `T`. The `simpRefls` are the action of the
generators on the basis of simple roots.
"""
mutable struct CoxGrpMat{T} <: CoxGrp
    # TODO: I originally typed this as cartanMat::MatElem{T} and simpRefls::Array{MatElem{T}, 1}, which
    # worked for the analagous Julia types (replacing MatElem{T} by Matrix{T}), but not for the Nemo
    # types. I wonder why...
    cartanMat::MatSpaceElem{T}
    simpRefls::Array{MatSpaceElem{T}, 1}
end

rank(grp::CoxGrpMat) = size(grp.cartanMat)[1]
gens(grp::CoxGrpMat) = [CoxEltMat(grp, refl, refl) for refl=grp.simpRefls]
function Base.one(grp::CoxGrpMat{T}) where {T}
    R = base_ring(grp.cartanMat)
    id = identity_matrix(R, rank(grp))
    return CoxEltMat(grp, id, id)
end

"""
Create a `CoxGrpMat`, a Coxeter group where elements are represented by matrices acting on the
basis of simple roots. The argument should be a Cartan matrix.
"""
function coxeter_group_mat(mat)
    cartanMat = to_cartan_matrix(mat)
    n = nrows(cartanMat)
    R = base_ring(cartanMat)

    # The simple reflection with index i acts on the weight space as
    #    s_i(x) = x - <x, alpha_i^> alpha_i
    # When x is written in the root basis, then <x, alpha_i^vee> is the
    # inner product of the entries of x with the ith row of the Cartan matrix.
    # In particular this takes the jth coordinate vector to e_j - a_ij e_i.
    simpRefls = [identity_matrix(R, n) for k=1:n]
    for i in 1:n
        for j in 1:n
            simpRefls[i][i, j] -= cartanMat[i, j]
        end
    end

    return CoxGrpMat(cartanMat, simpRefls)
end

"""
A CoxEltMat stores a Coxeter group element as a pair (mat, inv), where mat is the matrix of the
action in the basis of simple roots, and inv = mat^-1. The reason that inv is included is so that
we have fast access to the left descents as well as the right descents.
Ordinarily we would provide specialised implementations for many operators (eg there is an obvious
implementation of multiplication), but we are intentionally only providing a minimal implementation,
and falling back on the generic algorithms.
"""
struct CoxEltMat{T} <: CoxElt
    grp::CoxGrpMat{T}
    mat::MatSpaceElem{T}
    inv::MatSpaceElem{T}
end

Base.parent(x::CoxEltMat) = x.grp
Base.isequal(x::CoxEltMat, y::CoxEltMat) = x.grp == y.grp && x.mat == y.mat
Base.hash(x::CoxEltMat) = hash(x.grp, hash(x.mat))
Base.:(==)(x::CoxEltMat, y::CoxEltMat) = Base.isequal(x, y)
mult_gen(s::Integer, x::CoxEltMat) = CoxEltMat(x.grp, x.grp.simpRefls[s] * x.mat, x.inv * x.grp.simpRefls[s])
mult_gen(x::CoxEltMat, s::Integer) = CoxEltMat(x.grp, x.mat * x.grp.simpRefls[s], x.grp.simpRefls[s] * x.inv)

raw"""
`x.mat` represents the action of `x` on the simple root basis, hence the columns of the matrix are
``[w(\alpha_1) \cdots w(\alpha_r)]``. We have that ``ws < w`` if and only if ``w(\alpha_s) < 0``, and
negative roots are simple to distinguish in the simple root basis: they have at least one negative
coordinate (in fact all their coordinates will be negative or zero). Therefore the right descent set
is the set of columns which have at least one negative entry.
"""
right_descent_set(x::CoxEltMat) = BitSet(s for s=1:rank(x.grp) if any(x.mat[t, s] < 0 for t=1:rank(x.grp)))

raw"The left descents of ``w`` are the right descents of ``w^{-1}``."
left_descent_set(x::CoxEltMat) = BitSet(s for s=1:rank(x.grp) if any(x.inv[t, s] < 0 for t=1:rank(x.grp)))
end
