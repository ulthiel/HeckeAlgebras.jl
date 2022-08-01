using JuLie
include("coxeter_groups_MA.jl")
using Main.Coxeter_groups_MA
import AbstractAlgebra
import Nemo
import AbstractAlgebra:
    LaurentPolynomialRing, Generic.LaurentPolyWrapRing, Generic.LaurentPolyWrap, coeff, evaluate, change_base_ring, +, -, *, ^
import Base.*
import Base.getindex
import Base.:
import Base.one

export FreeModuleCox, HeckeAlgebra, AntisphericalModule, SphericalModule, BasisHecke, ElementHecke, :, getindex, one, -, +, *, ^, Bar, BasisHeckeStd, StandardBasis,
    HeckeAlgebraStdBasis, ASModuleStdBasis, SModuleStdBasis, BasisHeckeKL, KazhdanLusztigBasis, HeckeAlgebraKLBasis, ASModuleKLBasis, SModuleKLBasis

"""
    Reference
https://github.com/joelgibson/IHecke
"""

"""
    Coxeter groups
Additional functions for Coxeter groups.
"""
# Returns id of the Coxeter group.
function one(W::CoxGrp)
    id = identity_matrix(base_ring(W.cartanMat), rank(W))
    return CoxEltMat(W, id, id)
end

# Checks if the subset of W indexed by I and the left descent set of w are disjoint.
function IsMinimal(I::Array{Int, 1}, w::CoxElt)
    W = w.grp
    for i = 1:size(I)[1]
        s = W.simpRefls[i]
        if length(CoxEltMat(W, s, s) * w) < length(w)
            return false
        end
    end
    return true
end

"""
    FreeModuleCox
A **FreeModuleCox** is a free module over a commutative ring with basis elements indexed by (potentially a strict subset of) a Coxeter group.
"""
abstract type FreeModuleCox
end

"""
    HeckeAlgebra <: FreeModuleCox
A **HeckeAlgebra** is a free module over the Laurent polynomial ring over the integers with basis elements indexed by a Coxeter group.
"""
struct HeckeAlgebra <: FreeModuleCox
    BaseRing::LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing} # Commutative Ring this module is over. In case of the Hecke algebra it's the Laurent polynomial ring over the integers.
    CoxGroup::CoxGrp # Coxeter group of finitely presented type.
end

# Function for creating a new Hecke algebra.
function HeckeAlgebra(W::CoxGrp)
    A, v = LaurentPolynomialRing(ZZ, "v")
    return HeckeAlgebra(A, W)
end

"""
    AntisphericalModule <: FreeModuleCox
The **AntisphericalModule** is a module which is naturally associated to the Hecke algebra and a standard parabolic subgroup.
"""
struct AntisphericalModule <: FreeModuleCox
    BaseRing::LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing} # Commutative Ring this module is over. In case of the antispherical module it's the Laurent polynomial ring over the integers.
    CoxGroup::CoxGrp # Coxeter group of finitely presented type.
    Para::Array{Int, 1} # Parabolic subset of the Coxeter group.
end

# Function for creating a new antispherical module.
function AntisphericalModule(W::CoxGrp, I::Array{Int, 1})
    A, v = LaurentPolynomialRing(ZZ, "v")
    return AntisphericalModule(A, W, I)
end

"""
    SphericalModule <: FreeModuleCox
The **SphericalModule** is a module which is naturally associated to the Hecke algebra and a standard parabolic subgroup.
"""
struct SphericalModule <: FreeModuleCox
    BaseRing::LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing} # Commutative Ring this module is over. In case of the spherical module it's the Laurent polynomial ring over the integers.
    CoxGroup::CoxGrp # Coxeter group of finitely presented type.
    Para::Array{Int, 1} # Parabolic subset of the Coxeter group.
end

# Function for creating a new spherical module.
function SphericalModule(W::CoxGrp, I::Array{Int, 1})
    A, v = LaurentPolynomialRing(ZZ, "v")
    return SphericalModule(A, W, I)
end

"""
    BasisHecke
This type represents an object of type FreeModuleCox together with a choice of basis.
"""
abstract type BasisHecke
end

"""
    ElementHecke
An **ElementHecke** is a formal linear combination of Coxeter group elements, together with a **Parent** object which determines a basis.
"""
struct ElementHecke
    Parent::BasisHecke # A basis, i.e. an object of type BasisHecke.
    Terms::Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}
    # A dictionary mapping Coxeter group elements to scalars. No elements are mapped to zero.
end

# Fallback implementation. Bases should override either this, or the ElementHecke version.
function _ToBasis(A::BasisHecke, B::BasisHecke, w::CoxElt)
    return false
end

# Fallback implementation, which uses _ToBasis(::BasisHecke, ::BasisHecke, ::CoxElt) if available.
# Bases should override either this, or the CoxElt version.
function _ToBasis(A::BasisHecke, B::BasisHecke, eltB::ElementHecke)
    W = A.FreeModule.CoxGroup
    terms = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
    for w in keys(eltB.Terms)
        summand = _ToBasis(A, B, w)
        if summand == false
            return false
        end
        terms = _AddScaled(terms, summand.Terms, eltB.Terms[w])
    end
    return _ElementHeckeValidate(A, _RemoveZeros(ElementHecke(A, terms)))
end

# Changes eltB, which must be in the B basis, into the A basis.
function ToBasis(A::BasisHecke, B::BasisHecke, eltB::ElementHecke)
    A.FreeModule == B.FreeModule || throw(ArgumentError("A.FreeModule == B.FreeModule required"))
    B == eltB.Parent || throw(ArgumentError("B == eltB.Parent required"))

    # Simple case: If the bases are equal, nothing needs to be done.
    if typeof(A) == typeof(B)
        return eltB
    end

    # Is there a direct basis conversion from B to A defined?
    eltA = _ToBasis(A, B, eltB)
    if typeof(eltA) == ElementHecke
        A == eltA.Parent || throw(error(string("Basis conversion output ", eltA.Parent, " instead of ", A)))
        return eltA
    end

    # Convert via the standard basis of the free module.
    StdBasis = StandardBasis(A.FreeModule)
    eltStd = _ToBasis(StdBasis, B, eltB)
    typeof(eltStd) == ElementHecke || throw(error(string("No basis conversion defined for ", B, " into ", StdBasis)))
    eltStd.Parent == StdBasis || throw(error(string("Basis conversion output ", eltStd.Parent, " instead of ", StdBasis)))
    eltA = _ToBasis(A, StdBasis, eltStd)
    typeof(eltA) == ElementHecke || throw(error(string("No basis conversion defined for ", StdBasis, " into ", A)))
    eltA.Parent == A || throw(error(string("Basis conversion output ", eltA.Parent, " instead of ", A)))
    return eltA
end

# Shortcut for change of basis.
A::BasisHecke:elt::ElementHecke = ToBasis(A, elt.Parent, elt)

# Creates the basis element indexed by the Coxeter group element w.
function getindex(A::BasisHecke, w::CoxElt)
    w.grp == A.FreeModule.CoxGroup || throw(ArgumentError("w.grp == A.FreeModule.CoxGroup required"))
    terms = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
    terms[w] = one(A.FreeModule.BaseRing)
    return _ElementHeckeValidate(A, ElementHecke(A, terms))
end

# Creates the basis element indexed by the nth element of the generating set S of the Coxeter group of A.
function getindex(A::BasisHecke, n::Int)
    n > 0 || throw(ArgumentError("n > 0 required"))
    W = A.FreeModule.CoxGroup
    return A[CoxEltMat(W, W.simpRefls[n], W.simpRefls[n])]
end

# Creates the basis element indexed by an Array of integers.
function getindex(A::BasisHecke, L::Array{Int, 1})
    W = A.FreeModule.CoxGroup
    M = identity_matrix(base_ring(W.cartanMat), rank(W))
    for i in L
        M = M * W.simpRefls[i]
    end
    return A[CoxEltMat(W, M, M^-1)]
end

# Bases should override this function and throw an error if elt contains any illegal terms. For
# example, bases for the antispherical and spherical modules should reject non-minimal elements.
function _ElementHeckeValidate(B::BasisHecke, elt::ElementHecke)
    return elt
end

# Prints a Hecke algebra or module element in a regular form. Suppose the basis (and the basis symbol) is "H". Then,
# a scalar multiple of H[w] is printed as (scalar)H[w], with the identity element of the Coxeter
# group element being rendered as 'id'. Zero is printed as (0)H[id].
function Base.show(io::IO, mime::MIME"text/plain", elt::ElementHecke)
    symbol = elt.Parent.BasisSymbol
    W = elt.Parent.FreeModule.CoxGroup
    if elt.Terms == Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
        print("(0)*", symbol, "[id]")
    else
        bool = false
        for k in keys(elt.Terms)
            if bool
                print(" + ")
            end
            bool = true
            if Base.isone(k)
                print("(", elt.Terms[k], ")*", symbol, "[id]")
            else
                print("(", elt.Terms[k], ")*", symbol, lexword(k))
            end
        end
    end
end

# Removes all k in keys(elt.Terms) where elt.Terms[k] == 0.
function _RemoveZeros(elt::ElementHecke)
    Zero = zero(elt.Parent.FreeModule.BaseRing)
    terms = elt.Terms
    for k in keys(terms)
        if terms[k] == Zero
            terms = delete!(terms, k)
        end
    end
    return ElementHecke(elt.Parent, terms)
end

# Returns termsA + scalar * termsB.
function _AddScaled(termsA::Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}, termsB::Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}, scalar::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}})
    for w in keys(termsB)
        if w in keys(termsA)
            termsA[w] += termsB[w] * scalar
        else
            termsA[w] = termsB[w] * scalar
        end
    end
    return termsA
end

# Returns termsA + scalar * Basis[s].
function _AddScaledTerm(terms::Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}, w::CoxElt, scalar::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}})
    if w in keys(terms)
        terms[w] += scalar
    else
        terms[w] = scalar
    end
    return terms
end

# The negation of an element.
function -(elt::ElementHecke)
    terms = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
    for w in keys(elt.Terms)
        terms[w] = -elt.Terms[w]
    end
    P = elt.Parent
    return _ElementHeckeValidate(P, ElementHecke(P, terms))
end

# The sum of two elements.
function +(elt1::ElementHecke, elt2::ElementHecke)
    elt1.Parent.FreeModule == elt2.Parent.FreeModule || throw(error(string("Cannot add in different free modules ", elt1.Parent.FreeModule, " and ", elt2.Parent.FreeModule)))

    # Change into the left basis if necessary.
    if elt1.Parent != elt2.Parent
        elt2 = ToBasis(elt1.Parent, elt2.Parent, elt2)
    end
    assocs = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
    One = one(elt1.Parent.FreeModule.BaseRing)
    P = elt1.Parent
    return _ElementHeckeValidate(P, _RemoveZeros(ElementHecke(P, _AddScaled(_AddScaled(assocs, elt1.Terms, One), elt2.Terms, One))))
end

# The difference of two elements.
function -(elt1::ElementHecke, elt2::ElementHecke)
    elt1.Parent.FreeModule == elt2.Parent.FreeModule || throw(error(string("Cannot substract in different free modules ", elt1.Parent.FreeModule, " and ", elt2.Parent.FreeModule)))

    # Change into the left basis if necessary.
    if elt1.Parent != elt2.Parent
        elt2 = ToBasis(elt1.Parent, elt2.Parent, elt2)
    end
    assocs = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
    One = one(elt1.Parent.FreeModule.BaseRing)
    P = elt1.Parent
    return _ElementHeckeValidate(P, _RemoveZeros(ElementHecke(P, _AddScaled(_AddScaled(assocs, elt1.Terms, One), elt2.Terms, -One))))
end

# Multiplies an element by a scalar from the right.
function *(elt::ElementHecke, scalar::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}})
    assocs = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
    P = elt.Parent
    return _ElementHeckeValidate(P, _RemoveZeros(ElementHecke(P, _AddScaled(assocs, elt.Terms, scalar))))
end

# Multiplies an element by a scalar from the left.
function *(scalar::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, elt::ElementHecke)
    return elt * scalar
end

# Multiplies an element by an integer from the right.
function *(elt::ElementHecke, n::Int)
    return elt * (n * one(elt.Parent.FreeModule.BaseRing))
end

# Multiplies an element by an integer from the left.
function *(n::Int, elt::ElementHecke)
    return elt * (n * one(elt.Parent.FreeModule.BaseRing))
end

# The product of two elements.
function *(eltA::ElementHecke, eltB::ElementHecke)

    # In the case we are multiplying by zero, do nothing (might trigger a basis change otherwise).
    if eltA.Terms == Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}() || eltB.Terms == Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
        return ElementHecke(eltA.Parent, Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}())
    end

    # First check if there's a direct multiplication defined.
    result = _Multiply(eltA.Parent, eltA, eltB.Parent, eltB)
    if typeof(result) == ElementHecke
        return result
    end

    # Converts to standard basis and multiplies.
    stdA = StandardBasis(eltA.Parent.FreeModule)
    stdB = StandardBasis(eltB.Parent.FreeModule)
    product = _Multiply(stdA, ToBasis(stdA, eltA.Parent, eltA), stdB, ToBasis(stdB, eltB.Parent, eltB))
    typeof(product) == ElementHecke || throw(error(string("Multiplication not defined between ", eltA.Parent.FreeModule, " and ", eltB.Parent.FreeModule)))

    # Changes back to either the A or the B basis. Prefers the left one.
    if product.Parent.FreeModule == eltA.Parent.FreeModule
        return ToBasis(eltA.Parent, product.Parent, product)
    elseif product.Parent.FreeModule == eltB.Parent.FreeModule
        return ToBasis(eltB.Parent, product.Parent, product)
    end
    throw(error(string("Multiplication result in module ", product.Parent.FreeModule, " was incompatible with either of ", eltA.Parent.FreeModule, " or ", eltB.Parent.FreeModule)))
end

# Fallback implementation for multiplication.
function _Multiply(A::BasisHecke, eltA::ElementHecke, B::BasisHecke, eltB::ElementHecke)
    return false
end

# Exponantiation.
function ^(elt::ElementHecke, n::Int)
    n >= 0 || throw(ArgumentError(string("The exponent ", n, " must be nonnegative.")))
    if n == 0
        terms = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
        W = elt.Parent.FreeModule.CoxGroup
        terms[one(W)] = one(elt.Parent.FreeModule.BaseRing)
        return ElementHecke(elt.Parent, terms)
    end
    result = elt
    for k = 1:n-1
        result = result * elt
    end
    return result
end

# Needed for bar involution. Returns f(v^-1).
function twist(f::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}})
    A, v = LaurentPolynomialRing(ZZ, "v")
    return evaluate(change_base_ring(A, f), v^-1) # change_base_ring needed to get a datatype that works with evaluate.
end

# Perform the bar involution on elt, returning the result in the same basis as elt.
function Bar(elt::ElementHecke)
    # First check if there is a bar involution defined on this basis.
    result = _Bar(elt.Parent, elt)
    if typeof(result) == ElementHecke
        return result
    end

    # Convert to canonical basis, perform Bar, convert back.
    K = KazhdanLusztigBasis(elt.Parent.FreeModule)
    result = ToBasis(elt.Parent, K, Bar(ToBasis(K, elt.Parent, elt)))
    typeof(result) == ElementHecke || throw(error(string("Bar involution not defined on", K)))
    return result
end

# Fallback implementation of Bar involution.
function _Bar(A::BasisHecke, eltA::ElementHecke)
    return false
end

"""
    BasisHeckeStd <: BasisHecke
An abstract type, used for the standard basis of an object of type FreeModuleCox.
"""
abstract type BasisHeckeStd <: BasisHecke
end

# (M, eltM) is an element inside a right module over the Hecke algebra (either the algebra itself,
# or a parabolic module). Returns the right action of H[s] on eltM.
function _RightMultStdGen(M::BasisHeckeStd, eltM::ElementHecke, s::CoxElt)
    W = M.FreeModule.CoxGroup
    v = gen(M.FreeModule.BaseRing)
    terms = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
    for w in keys(eltM.Terms)
        ws = w * s
        if ! IsMinimal(M.Para, ws)
            _AddScaledTerm(terms, w, M.Eig * eltM.Terms[w])
        elseif length(w) < length(ws)
            _AddScaledTerm(terms, ws, eltM.Terms[w])
        else
            _AddScaledTerm(terms, ws, eltM.Terms[w])
            _AddScaledTerm(terms, w, (v^-1 - v) * eltM.Terms[w])
        end
    end
    return _ElementHeckeValidate(M, _RemoveZeros(ElementHecke(M, terms)))
end

# (M, eltM) is an element inside a right module over the Hecke algebra (either the algebra itself,
# or a parabolic module). This function returns eltM * H[terms], where terms is interpreted as a
# linear combination of standard basis elements of the Hecke algebra.
function _RightAction(M::BasisHeckeStd, eltM::ElementHecke, terms::Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}})
    acc = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
    W = M.FreeModule.CoxGroup
    for w in keys(terms)
        piece = eltM
        for i in lexword(w)
            s = W.simpRefls[i]
            piece = _RightMultStdGen(M, piece, CoxEltMat(W, s, s))
        end
        _AddScaled(acc, piece.Terms, terms[w])
    end
    return _ElementHeckeValidate(M, _RemoveZeros(ElementHecke(M, acc)))
end

# Return the bar involution of the basis element M[w], i.e. the image of the element H[w^-1]^-1
# inside the module M. (Pick a right descent s, and use H[w^-1]^-1 = H[ws^-1]^-1 * H[s]^-1).
function _BarInvolutionStd(M::BasisHeckeStd, w::CoxElt)
    if w in keys(M.BarCache)
        return M.BarCache[w]
    end
    W = w.grp
    if length(w) == 0
        return M[one(W)]
    end
    v = gen(M.FreeModule.BaseRing)
    m = W.simpRefls[first(right_descent_set(w))]
    s = CoxEltMat(W, m, m)
    bar_ws = _BarInvolutionStd(M, w*s)
    bar_w = _RightMultStdGen(M, bar_ws, s) + (v - v^-1) * bar_ws
    M.BarCache[w] = bar_w
    return bar_w
end

# The bar involution on elt, mapping p(v)H[w] to p(v^-1)H[w^-1]^-1.
function _Bar(H::BasisHeckeStd, elt::ElementHecke)
    H == elt.Parent || throw(ArgumentError("H == elt.Parent required"))
    R = H.FreeModule.BaseRing
    terms = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
    for w in keys(elt.Terms)
        _AddScaled(terms, _BarInvolutionStd(H, w).Terms, twist(elt.Terms[w]))
    end
    return _ElementHeckeValidate(H, _RemoveZeros(ElementHecke(H, terms)))
end

"""
    HeckeAlgebraStdBasis <: BasisHeckeStd
A type, used for the standard basis of the Hecke algebra.
"""
struct HeckeAlgebraStdBasis <: BasisHeckeStd
    FreeModule::HeckeAlgebra # The Hecke algebra this basis belongs to.
    BasisSymbol::Union{AbstractString, Char, Symbol}  # A short string naming the basis. In case of the standard basis it's "H".
    BarCache::Dict{CoxElt, ElementHecke} # A dictionary, used as a cache for the bar involution on basis elements.
    Para::Array{Int, 1} # Parabolic subset (equals Array{Int, 1}() for the Hecke algebra).
    Eig::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}
    # Irrelevant for the full Hecke algebra , -v or v^-1 for the antispherical/spherical modules.
end

# Creates the standard basis of the Hecke algebra.
StandardBasis(HAlg::HeckeAlgebra) = HeckeAlgebraStdBasis(HAlg, "H", Dict{CoxElt, ElementHecke}(), Array{Int, 1}(), zero(HAlg.BaseRing))

# Multiplication inside the standard basis of the Hecke algebra.
function _Multiply(H1::HeckeAlgebraStdBasis, elt1::ElementHecke, H2::HeckeAlgebraStdBasis, elt2::ElementHecke)
    return _RightAction(H1, elt1, elt2.Terms)
end

"""
    ASModuleStdBasis <: BasisHeckeStd
A type, used for the standard basis of the antispherical module.
"""
struct ASModuleStdBasis <: BasisHeckeStd
    FreeModule::AntisphericalModule # The antispherical module this basis belongs to.
    BasisSymbol::Union{AbstractString, Char, Symbol}  # A short string naming the basis. In case of the standard basis it's "aH".
    BarCache::Dict{CoxElt, ElementHecke} # A dictionary, used as a cache for the bar involution on basis elements.
    Para::Array{Int, 1} # Parabolic subset (equals Array{Int, 1}() for the Hecke algebra).
    Eig::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}
    # Irrelevant for the full Hecke algebra , -v or v^-1 for the antispherical/spherical modules.
end

# Creates the standard basis of the antispherical module.
StandardBasis(ASMod::AntisphericalModule) = ASModuleStdBasis(ASMod, "aH", Dict{CoxElt, ElementHecke}(), ASMod.Para, -gen(ASMod.BaseRing))

# Only allow I-minimal elements.
function _ElementHeckeValidate(aH::ASModuleStdBasis, elt::ElementHecke)
    I = aH.Para
    for w in keys(elt.Terms)
        IsMinimal(I, w) || throw(error(string(w, " is not minimal with respect to ", aH.FreeModule.Para)))
    end
    return elt
end

# Multiplication inside the standard basis of the antispherical module.
function _Multiply(M::ASModuleStdBasis, eltM::ElementHecke, H::HeckeAlgebraStdBasis, eltH::ElementHecke)
    return _RightAction(M, eltM, eltH.Terms)
end

"""
    SModuleStdBasis <: BasisHeckeStd
A type, used for the standard basis of the spherical module.
"""
struct SModuleStdBasis <: BasisHeckeStd
    FreeModule::SphericalModule # The spherical module this basis belongs to.
    BasisSymbol::Union{AbstractString, Char, Symbol}  # A short string naming the basis. In case of the standard basis it's "sH".
    BarCache::Dict{CoxElt, ElementHecke} # A dictionary, used as a cache for the bar involution on basis elements.
    Para::Array{Int, 1} # Parabolic subset (equals Array{Int, 1}() for the Hecke algebra).
    Eig::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}
    # Irrelevant for the full Hecke algebra , -v or v^-1 for the antispherical/spherical modules.
end

# Creates the standard basis of the spherical module.
StandardBasis(SMod::SphericalModule) = SModuleStdBasis(SMod, "sH", Dict{CoxElt, ElementHecke}(), SMod.Para, gen(SMod.BaseRing)^-1)

# Only allow I-minimal elements.
function _ElementHeckeValidate(sH::SModuleStdBasis, elt::ElementHecke)
    I = sH.Para
    for w in keys(elt.Terms)
        IsMinimal(I, w) || throw(error(string(w, " is not minimal with respect to sH.FreeModule.Para")))
    end
    return elt
end

# Multiplication inside the standard basis of the spherical module.
function _Multiply(M::SModuleStdBasis, eltM::ElementHecke, H::HeckeAlgebraStdBasis, eltH::ElementHecke)
    return _RightAction(M, eltM, eltH.Terms)
end

"""
    BasisHeckeKL <: BasisHecke
An abstract type, used for the Kazhdan-Lusztig basis of an object of type FreeModuleCox.
"""
abstract type BasisHeckeKL <: BasisHecke
end

# Assume that elt is an element written in terms of the standard basis, and right-multiply
# by the element B[s]. I is the parabolic quotient (Array{Int, 1}() if we are in the Hecke algebra), and eig
# is -v for antispherical, v^-1 for spherical, and irrelevant if in the Hecke algebra.
function _RightMultKLGen(elt::ElementHecke, s::CoxElt, I::Array{Int, 1}, eig::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}})
    W = elt.Parent.FreeModule.CoxGroup
    v = gen(elt.Parent.FreeModule.BaseRing)
    terms = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
    for w in keys(elt.Terms)
        ws = w * s
        if !(length(I) == 0) && !(IsMinimal(I, ws))
            terms = elt.Parent, _AddScaledTerm(terms, w, (v + eig) * elt.Terms[w])
        elseif length(w) < length(ws)
            terms = _AddScaledTerm(terms, ws, elt.Terms[w])
            terms = _AddScaledTerm(terms, w, v * elt.Terms[w])
        else
            terms = _AddScaledTerm(terms, ws, elt.Terms[w])
            terms = _AddScaledTerm(terms, w, v^-1 * elt.Terms[w])
        end
    end
    P = elt.Parent
    return _ElementHeckeValidate(P, _RemoveZeros(ElementHecke(P, terms)))
end

# A dictionary mapping group elements u to the coefficient mu(u, w).
function _MuCoeffs(H::BasisHeckeStd, K::BasisHeckeKL, w::CoxElt)
    if w in keys(K.MuCache)
        return K.MuCache[w]
    end
    if length(w) == 0
        return Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
    end
    Cw = _ToBasis(H, K, w)
    mu = Dict{CoxElt, Nemo.fmpz}()
    for u in keys(Cw.Terms)
        x_coeff = coeff(Cw.Terms[u], 1)
        if x_coeff != zero(K.FreeModule.BaseRing)
            mu[u] = x_coeff
        end
    end
    K.MuCache[w] = mu
    return mu
end

# Express K[w] in the standard basis.
function _KLToStd(H::BasisHeckeStd, K::BasisHeckeKL, w::CoxElt, I::Array{Int, 1}, eig::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}})
    if w in keys(K.KLInStdCache)
        return K.KLInStdCache[w]
    end
    W = H.FreeModule.CoxGroup
    if length(w) == 0
        id = identity_matrix(base_ring(W.cartanMat), rank(W))
        return H[CoxEltMat(W, id, id)]
    end

    # Take an arbitrary right descent of w.
    m = W.simpRefls[first(right_descent_set(w))]
    s = CoxEltMat(W, m, m)
    ws = w * s

    # K[ws] = K[w]K[s] - sum[us < u]mu(u, ws)K[u]
    Kw = _ToBasis(H, K, ws)
    KwKs = _RightMultKLGen(Kw, s, I, eig)
    for u in keys(_MuCoeffs(H, K, ws))
        if length(u * s) < length(u)
            KwKs -= _MuCoeffs(H, K, ws)[u] * _ToBasis(H, K, u)
        end
    end
    K.KLInStdCache[w] = KwKs
    return KwKs
end

# Express H[w] in the Kazhdan-Lusztig basis.
function _StdToKL(H::BasisHeckeStd, K::BasisHeckeKL, w::CoxElt)
    if w in keys(K.StdInKLCache)
        return K.StdInKLCache[w]
    end
    W = H.FreeModule.CoxGroup
    if length(w) == 0
        id = identity_matrix(base_ring(W.cartanMat), rank(W))
        return K[CoxEltMat(W, id, id)]
    end
    terms = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
    terms[w] = one(K.FreeModule.BaseRing)
    Kw = _ToBasis(H, K, w)
    for u in keys(Kw.Terms)
        if u != w
            Ku = _ToBasis(K, H, u)
            terms = _AddScaled(terms, Ku.Terms, -Kw.Terms[u])
        end
    end
    result = _ElementHeckeValidate(K, _RemoveZeros(ElementHecke(K, terms)))
    K.StdInKLCache[w] = result
    return result
end

# The bar involution of elt, fixing each basis element and twisting scalars by v -> v^-1.
function _Bar(K::BasisHeckeKL, elt::ElementHecke)
    K == elt.Parent || throw(ArgumentError("K == elt.Parent required"))
    terms = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
    for w in keys(elt.Terms)
        terms[w] = twist(elt.Terms[w])
    end
    return _ElementHeckeValidate(K, ElementHecke(K, terms))
end

"""
    HeckeAlgebraKLBasis <: BasisHeckeKL
A type, used for the Kazhdan-Lusztig basis of the Hecke algebra.
"""
struct HeckeAlgebraKLBasis <: BasisHeckeKL
    FreeModule::HeckeAlgebra # The Hecke algebra this basis belongs to.
    BasisSymbol::Union{AbstractString, Char, Symbol} # A short string naming the basis. In case of the Kazhdan-Lusztig basis it's "K".
    KLInStdCache::Dict{CoxElt, ElementHecke} # A dictionary, used as a cache for the basis change from Kazhdan-Lusztig basis to standard basis.
    StdInKLCache::Dict{CoxElt, ElementHecke} # A dictionary, used as a cache for the basis change from standard basis to Kazhdan-Lusztig basis.
    MuCache::Dict{CoxElt, Dict{CoxElt, Nemo.fmpz}} # A dictionary, used as a cache for the coefficient mu.
    Para::Array{Int, 1} # Parabolic subset (equals Array{Int, 1}() for the Hecke algebra).
    Eig::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}
    # Irrelevant for the full Hecke algebra , -v or v^-1 for the antispherical/spherical modules.
end

# Creates the Kazhdan-Lusztig basis of the Hecke algebra.
KazhdanLusztigBasis(HAlg::HeckeAlgebra) = HeckeAlgebraKLBasis(HAlg, "K", Dict{CoxElt, ElementHecke}(), Dict{CoxElt, ElementHecke}(), Dict{CoxElt, Dict{CoxElt, Nemo.fmpz}}(), Array{Int, 1}(), zero(HAlg.BaseRing))

# Express K[w] in the standard basis.
function _ToBasis(H::HeckeAlgebraStdBasis, K::HeckeAlgebraKLBasis, w::CoxElt)
    return _KLToStd(H, K, w, Array{Int, 1}(), zero(H.FreeModule.BaseRing))
end

# Express H[w] in the Kazhdan-Lusztig basis.
function _ToBasis(K::HeckeAlgebraKLBasis, H::HeckeAlgebraStdBasis, w::CoxElt)
    return _StdToKL(H, K, w)
end

# Intercept left or right multiplication by the identity or K[s], and implement it in terms
# of the mu-coefficients. Otherwise return false (defers to standard multiplication).
function _Multiply(K1::HeckeAlgebraKLBasis, eltA::ElementHecke, K2::HeckeAlgebraKLBasis, eltB::ElementHecke)
    K1 == K2 || throw(ArgumentError("K1 == K2 required"))
    H = StandardBasis(K1.FreeModule)
    W = K1.FreeModule.CoxGroup
    v = gen(K1.FreeModule.BaseRing)
    # Two special cases: multiplication by the identity, or by K[s] for a simple generator s.
    # For multiplication by K[s] we have the formulas
    #   K[s] * K[w] = (v + v^-1)K[w] if sw < w.
    #   K[s] * K[w] = K[sw] + sum[z | sz < z]mu(z, w)K[z].

    if length(eltA.Terms) == 1
        for s in keys(eltA.Terms)
            scale = eltA.Terms[s]
            if length(s) == 0
                return scale * eltB
            end
            if length(s) == 1
                terms = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
                for w in keys(eltB.Terms)
                    coeff = eltB.Terms[w]
                    if length(s * w) < length(w)
                        terms = _AddScaledTerm(terms, w, scale * coeff * (v + v^-1))
                    else
                        terms = _AddScaledTerm(terms, s * w, scale * coeff)
                        mu = _MuCoeffs(H, K1, w)
                        for z in keys(mu)
                            if length(s * z) < length(z)
                                terms = _AddScaledTerm(terms, z, mu[z] * scale * coeff)
                            end
                        end
                    end
                end
                return _ElementHeckeValidate(K1, _RemoveZeros(ElementHecke(K1, terms)))
            end
        end
    end
    if length(eltB.Terms) == 1
        for s in keys(eltB.Terms)
            scale = eltB.Terms[s]
            if length(s) == 0
                return scale * eltA
            end
            if length(s) == 1
                terms = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
                for w in keys(eltA.Terms)
                    coeff = eltA.Terms[w]
                    if length(w * s) < length(w)
                        terms = _AddScaledTerm(terms, w, scale * coeff * (v + v^-1))
                    else
                        terms = _AddScaledTerm(terms, w * s, scale * coeff)
                        mu = _MuCoeffs(H, K1, w)
                        for z in keys(mu)
                            if length(z * s) < length(z)
                                terms = _AddScaledTerm(terms, z, mu[z] * scale * coeff)
                            end
                        end
                    end
                end
                return _ElementHeckeValidate(K1, _RemoveZeros(ElementHecke(K1, terms)))
            end
        end
    end
    return false
end

"""
    ASModuleKLBasis <: BasisHeckeKL
A type, used for the Kazhdan-Lusztig basis of the antispherical module.
"""
struct ASModuleKLBasis <: BasisHeckeKL
    FreeModule::AntisphericalModule # The antispherical module this basis belongs to.
    BasisSymbol::Union{AbstractString, Char, Symbol} # A short string naming the basis. In case of the Kazhdan-Lusztig basis it's "aK".
    KLInStdCache::Dict{CoxElt, ElementHecke} # A dictionary, used as a cache for the basis change from Kazhdan-Lusztig basis to standard basis.
    StdInKLCache::Dict{CoxElt, ElementHecke} # A dictionary, used as a cache for the basis change from standard basis to Kazhdan-Lusztig basis.
    MuCache::Dict{CoxElt, Dict{CoxElt, Nemo.fmpz}} # A dictionary, used as a cache for the coefficient mu.
    Para::Array{Int, 1} # Parabolic subset (equals Array{Int, 1}() for the Hecke algebra).
    Eig::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}
    # Irrelevant for the full Hecke algebra , -v or v^-1 for the antispherical/spherical modules.
end

# Creates the Kazhdan-Lusztig basis of the antispherical module.
KazhdanLusztigBasis(ASMod::AntisphericalModule) = ASModuleKLBasis(ASMod, "aK", Dict{CoxElt, ElementHecke}(), Dict{CoxElt, ElementHecke}(), Dict{CoxElt, Dict{CoxElt, Nemo.fmpz}}(), ASMod.Para, -gen(ASMod.BaseRing))

# Only allow I-minimal elements.
function _ElementHeckeValidate(aK::ASModuleKLBasis, elt::ElementHecke)
    I = aK.Para
    for w in keys(elt.Terms)
        IsMinimal(I, w) || throw(error(string(w, " is not minimal with respect to ", aH.FreeModule.Para)))
    end
    return elt
end

# Express aK[w] in the standard basis.
function _ToBasis(aH::ASModuleStdBasis, aK::ASModuleKLBasis, w::CoxElt)
    return _KLToStd(aH, aK, w, aK.Para, aH.Eig)
end

# Express aH[w] in the Kazhdan-Lusztig basis.
function _ToBasis(aK::ASModuleKLBasis, aH::ASModuleStdBasis, w::CoxElt)
    return _StdToKL(aH, aK, w)
end

"""
    SModuleKLBasis <: BasisHeckeKL
A type, used for the Kazhdan-Lusztig basis of the spherical module.
"""
struct SModuleKLBasis <: BasisHeckeKL
    FreeModule::SphericalModule # The spherical module this basis belongs to.
    BasisSymbol::Union{AbstractString, Char, Symbol} # A short string naming the basis. In case of the Kazhdan-Lusztig basis it's "sK".
    KLInStdCache::Dict{CoxElt, ElementHecke} # A dictionary, used as a cache for the basis change from Kazhdan-Lusztig basis to standard basis.
    StdInKLCache::Dict{CoxElt, ElementHecke} # A dictionary, used as a cache for the basis change from standard basis to Kazhdan-Lusztig basis.
    MuCache::Dict{CoxElt, Dict{CoxElt, Nemo.fmpz}} # A dictionary, used as a cache for the coefficient mu.
    Para::Array{Int, 1} # Parabolic subset (equals Array{Int, 1}() for the Hecke algebra).
    Eig::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}
    # Irrelevant for the full Hecke algebra , -v or v^-1 for the antispherical/spherical modules.
end

# Creates the Kazhdan-Lusztig basis of the spherical module.
KazhdanLusztigBasis(SMod::SphericalModule) = SModuleKLBasis(SMod, "sK", Dict{CoxElt, ElementHecke}(), Dict{CoxElt, ElementHecke}(), Dict{CoxElt, Dict{CoxElt, Nemo.fmpz}}(), SMod.Para, gen(SMod.BaseRing)^-1)

# Only allow I-minimal elements.
function _ElementHeckeValidate(sK::SModuleKLBasis, elt::ElementHecke)
    I = sK.Para
    for w in keys(elt.Terms)
        IsMinimal(I, w) || throw(error(string(w, " is not minimal with respect to ", sH.FreeModule.Para)))
    end
    return elt
end

# Express sK[w] in the standard basis.
function _ToBasis(sH::SModuleStdBasis, sK::SModuleKLBasis, w::CoxElt)
    return _KLToStd(sH, sK, w, sK.Para, sH.Eig)
end

# Express sH[w] in the Kazhdan-Lusztig basis.
function _ToBasis(sK::SModuleKLBasis, sH::SModuleStdBasis, w::CoxElt)
    return _StdToKL(sH, sK, w)
end

W = Main.coxeter_group_mat([1 3 2; 3 1 4; 2 4 1])
HAlg = HeckeAlgebra(W)
H = StandardBasis(HAlg)
K = KazhdanLusztigBasis(HAlg)
A, v = LaurentPolynomialRing(ZZ, "v")

H[1]
H[3]*H[1]
H[3]*K[2]
H[3]*(H[1] + v*H[[1, 2, 3]])
2*H[3]*K[2]^3 + (v^2 + v^-1)*K[1]
Bar(K[1])
Bar(H[3])
H:K[2]
ASModule = AntisphericalModule(W, [1, 2])
aH = StandardBasis(ASModule)
aK = KazhdanLusztigBasis(ASModule)
Bar(aH[3])
aH[3]*H[3]
SModule = SphericalModule(W, [1, 3])
sH = StandardBasis(SModule)
sK = KazhdanLusztigBasis(SModule)
sH[3]*H[3]