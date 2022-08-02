# By M. Albert (2022)
# Implementierung der Hecke Algebra, die auf coxeter_groups_MA aufbaut und einen allgemeinen Teil für beliebige freie Moduln besitzt.
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

export FreeMod, FreeModuleBasis, FreeModuleElement, DefaultBasis, getindex, HeckeAlgebra, AntisphericalModule, SphericalModule, :, one, -, +, *, ^, Bar, BasisHeckeStd, StandardBasis, HeckeAlgebraStdBasis,
    ASModuleStdBasis, SModuleStdBasis, BasisHeckeKL, KazhdanLusztigBasis, HeckeAlgebraKLBasis, ASModuleKLBasis, SModuleKLBasis

"""
    Introduction
The following is an implementation of arbitrary free modules i.e. free modules over an arbitrary ring with an arbitrary set as it's index set.

Reference: https://github.com/HechtiDerLachs/Oscar.jl/blob/homological_algebra/src/HomologicalAlgebra/InfiniteDirectSums.jl

To use this interface the following needs to be done:

1.: Construct a new free module type i.e. a type inherting from FreeMod{BaseRingType, BaseRingElemType, IndexSetType, IndexType}.
    It should at least have the two fields BaseRing::BaseRingType, IndexSet::IndexSetType. For example:

    struct ABCModule <: FreeMod{Nemo.FlintIntegerRing, Nemo.fmpz, Array{String, 1}, String}
        BaseRing::Nemo.FlintIntegerRing
        IndexSet::Array{String, 1}
    end

    ABCModule(A::Array{String, 1}) = ABCModule(ZZ, A)

2.: Construct at least one new basis type for your free module type i.e a type inherting from FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}.
    Each of these types should at least have the two fields FreeModule::T <: FreeMod{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, BasisSymbol::Union{AbstractString, Char, Symbol}.
    For example:

    struct ABCModuleBasis <: FreeModuleBasis{Nemo.FlintIntegerRing, Nemo.fmpz, Array{String, 1}, String}
        FreeModule::ABCModule
        BasisSymbol::Union{AbstractString, Char, Symbol}
    end

    ABCModuleBasis(ABCMod::ABCModule) = ABCModuleBasis(ABCMod, "X")

    struct ABCModuleBasis2 <: FreeModuleBasis{Nemo.FlintIntegerRing, Nemo.fmpz, Array{String, 1}, String}
        FreeModule::ABCModule
        BasisSymbol::Union{AbstractString, Char, Symbol}
    end

    ABCModuleBasis2(ABCMod::ABCModule) = ABCModuleBasis2(ABCMod, "Y")

3.: Declare which of your basis should be the default basis like this:

    DefaultBasis(ABCMod::ABCModule) = ABCModuleBasis(ABCMod)

4.: Define the multiplication in the default basis. (It can be defined for the other bases as well but that's not necessary.)
    To do this you need to overwrite the function _Multiply for your default basis type like this:

    function _Multiply(A::ABCModuleBasis, eltA::FreeModuleElement{Nemo.FlintIntegerRing, Nemo.fmpz, Array{String, 1}, String}, B::ABCModuleBasis, eltB::FreeModuleElement{Nemo.FlintIntegerRing, Nemo.fmpz, Array{String, 1}, String})
        terms = Dict{String, Nemo.fmpz}()
        terms["a"] = terms["b"] = terms["c"] = 0
        for w in keys(eltA.Terms)
            scalarA = eltA.Terms[w]
            for s in keys(eltB.Terms)
                result = scalarA * eltB.Terms[s]
                if (w == "a" && s == "a") || (w == "b" && s == "c") || (w == "c" && s == "b")
                    terms["a"] = terms["a"] + result
                elseif (w == "a" && s == "b") || (w == "b" && s == "a") || (w == "c" && s == "c")
                    terms["b"] = terms["b"] + result
                else
                    terms["c"] = terms["c"] + result
                end
            end
        end
        return FreeModuleElement{Nemo.FlintIntegerRing, Nemo.fmpz, Array{String, 1}, String}(A, terms)
    end

5.: For every basis A except the default basis define a basis convertion from A to the default basis and back.
    To do this you need to overwrite the function _ToBasis(A::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, B::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, w::IndexType) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    or the function _ToBasis(A::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, B::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, eltB::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}.
    For example:

    function _ToBasis(A::ABCModuleBasis, B::ABCModuleBasis2, w::String)
        if w == "a"
            return A["a"]
        elseif w == "b"
            return A["a"] + A["b"]
        else
            return A["a"] + A["b"] + A["c"]
        end
    end

    function _ToBasis(B::ABCModuleBasis2, A::ABCModuleBasis, w::String)
        if w == "a"
            return B["a"]
        elseif w == "b"
            return B["b"] - B["a"]
        else
            return B["c"] - B["a"]
        end
    end
"""

"""
    FreeMod{BaseRingType, BaseRingElemType, IndexSetType, IndexType}
An abstract type, used for an arbitrary free module. A type inherting from FreeMod{BaseRingType, BaseRingElemType, IndexSetType, IndexType}
should at least have the fields BaseRing::BaseRingType, IndexSet::IndexSetType.
"""
abstract type FreeMod{BaseRingType, BaseRingElemType, IndexSetType, IndexType}
end

"""
    FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}
An abstract type, used for a basis of an arbitrary free module. A type inherting from FreeModBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}
should at least have the fields FreeModule::T <: FreeMod{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, BasisSymbol::Union{AbstractString, Char, Symbol}.
"""
abstract type FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}
end

# Fallback implementation of default basis. For every free module a default basis needs to be defined.
function DefaultBasis(FMod::FreeMod{BaseRingType, BaseRingElemType, IndexSetType, IndexType}) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    throw(error(string("No default basis defined for ", FMod)))
end

"""
    FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}
A concrete type, used for an element of an arbitrary free module in a given basis.
"""
struct FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    Parent::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    Terms::Dict{IndexType, BaseRingElemType}
end

# Fallback implementation. Bases should override either this, or the FreeModuleElement version.
function _ToBasis(A::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, B::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, w::IndexType) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    return false
end

# Fallback implementation, which uses _ToBasis(A::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, B::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, w::IndexType)
# if available. Bases should override either this, or the IndexType version.
function _ToBasis(A::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, B::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, eltB::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    W = A.FreeModule.IndexSet
    terms = Dict{IndexType, BaseRingElemType}()
    for w in keys(eltB.Terms)
        summand = _ToBasis(A, B, w)
        if summand == false
            return false
        end
        terms = _AddScaled(terms, summand.Terms, eltB.Terms[w])
    end
    return _FreeModuleElementValidate(A, _RemoveZeros(FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}(A, terms)))
end

# Changes eltB, which must be in the B basis, into the A basis.
function ToBasis(A::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, B::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, eltB::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    A.FreeModule == B.FreeModule || throw(ArgumentError("A.FreeModule == B.FreeModule required"))
    B == eltB.Parent || throw(ArgumentError("B == eltB.Parent required"))

    # Simple case: If the bases are equal, nothing needs to be done.
    if typeof(A) == typeof(B)
        return eltB
    end

    # Is there a direct basis conversion from B to A defined?
    eltA = _ToBasis(A, B, eltB)
    if typeof(eltA) == FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}
        A == eltA.Parent || throw(error(string("Basis conversion output ", eltA.Parent, " instead of ", A)))
        return eltA
    end

    # Convert via the default basis of the free module.
    DefBasis = DefaultBasis(A.FreeModule)
    eltDef = _ToBasis(DefBasis, B, eltB)
    typeof(eltDef) == FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType} || throw(error(string("No basis conversion defined for ", B, " into ", DefBasis)))
    eltDef.Parent == DefBasis || throw(error(string("Basis conversion output ", eltDef.Parent, " instead of ", DefBasis)))
    eltA = _ToBasis(A, DefBasis, eltDef)
    typeof(eltA) == FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType} || throw(error(string("No basis conversion defined for ", DefBasis, " into ", A)))
    eltA.Parent == A || throw(error(string("Basis conversion output ", eltA.Parent, " instead of ", A)))
    return eltA
end

# Shortcut for change of basis.
A::FreeModuleBasis:elt::FreeModuleElement = ToBasis(A, elt.Parent, elt)

# Creates the basis element indexed by the index set element w.
function getindex(A::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, w::IndexType) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    terms = Dict{IndexType, BaseRingElemType}()
    terms[w] = one(A.FreeModule.BaseRing)
    return _FreeModuleElementValidate(A, FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}(A, terms))
end

# Bases should override this function and throw an error if elt contains any illegal terms. For
# example, bases for the antispherical and spherical modules should reject non-minimal elements.
function _FreeModuleElementValidate(B::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, elt::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    return elt
end

# Prints a free module element in a regular form. Suppose the basis (and the basis symbol) is "H". Then,
# a scalar multiple of H[w] is printed as (scalar)H[w]. Zero is printed as (0).
function Base.show(io::IO, mime::MIME"text/plain", elt::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    symbol = elt.Parent.BasisSymbol
    W = elt.Parent.FreeModule.IndexSet
    if elt.Terms == Dict{IndexType, BaseRingElemType}()
        print("(0)")
    else
        bool = false
        for k in keys(elt.Terms)
            if bool
                print(" + ")
            end
            bool = true
            print("(", elt.Terms[k], ")*", symbol)
            show(k)
        end
    end
end

# Removes all k in keys(elt.Terms) where elt.Terms[k] == 0.
function _RemoveZeros(elt::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    Zero = zero(elt.Parent.FreeModule.BaseRing)
    terms = elt.Terms
    for k in keys(terms)
        if terms[k] == Zero
            terms = delete!(terms, k)
        end
    end
    return FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}(elt.Parent, terms)
end

# Returns termsA + scalar * termsB.
function _AddScaled(termsA, termsB, scalar)
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
function _AddScaledTerm(terms, w, scalar)
    if w in keys(terms)
        terms[w] += scalar
    else
        terms[w] = scalar
    end
    return terms
end

# The negation of an element.
function -(elt::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    terms = Dict{IndexType, BaseRingElemType}()
    for w in keys(elt.Terms)
        terms[w] = -elt.Terms[w]
    end
    P = elt.Parent
    return _FreeModuleElementValidate(P, FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}(P, terms))
end

# The sum of two elements.
function +(elt1::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, elt2::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    elt1.Parent.FreeModule == elt2.Parent.FreeModule || throw(error(string("Cannot add in different free modules ", elt1.Parent.FreeModule, " and ", elt2.Parent.FreeModule)))

    # Change into the left basis if necessary.
    if elt1.Parent != elt2.Parent
        elt2 = ToBasis(elt1.Parent, elt2.Parent, elt2)
    end
    assocs = Dict{IndexType, BaseRingElemType}()
    One = one(elt1.Parent.FreeModule.BaseRing)
    P = elt1.Parent
    return _FreeModuleElementValidate(P, _RemoveZeros(FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}(P, _AddScaled(_AddScaled(assocs, elt1.Terms, One), elt2.Terms, One))))
end

# The difference of two elements.
function -(elt1::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, elt2::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    elt1.Parent.FreeModule == elt2.Parent.FreeModule || throw(error(string("Cannot substract in different free modules ", elt1.Parent.FreeModule, " and ", elt2.Parent.FreeModule)))

    # Change into the left basis if necessary.
    if elt1.Parent != elt2.Parent
        elt2 = ToBasis(elt1.Parent, elt2.Parent, elt2)
    end
    assocs = Dict{IndexType, BaseRingElemType}()
    One = one(elt1.Parent.FreeModule.BaseRing)
    P = elt1.Parent
    return _FreeModuleElementValidate(P, _RemoveZeros(FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}(P, _AddScaled(_AddScaled(assocs, elt1.Terms, One), elt2.Terms, -One))))
end

# Multiplies an element by a scalar from the right.
function *(elt::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, scalar::BaseRingElemType) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    assocs = Dict{IndexType, BaseRingElemType}()
    P = elt.Parent
    return _FreeModuleElementValidate(P, _RemoveZeros(FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}(P, _AddScaled(assocs, elt.Terms, scalar))))
end

# Multiplies an element by a scalar from the left.
function *(scalar::BaseRingElemType, elt::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    return elt * scalar
end

# Multiplies an element by an integer from the right.
function *(elt::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, n::Int) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    return elt * (n * one(elt.Parent.FreeModule.BaseRing))
end

# Multiplies an element by an integer from the left.
function *(n::Int, elt::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    return elt * (n * one(elt.Parent.FreeModule.BaseRing))
end

# The product of two elements.
function *(eltA::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, eltB::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}

    # In the case we are multiplying by zero, do nothing (might trigger a basis change otherwise).
    if eltA.Terms == Dict{IndexType, BaseRingElemType}() || eltB.Terms == Dict{IndexType, BaseRingElemType}()
        return FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}(eltA.Parent, Dict{IndexType, BaseRingElemType}())
    end

    # First check if there's a direct multiplication defined.
    result = _Multiply(eltA.Parent, eltA, eltB.Parent, eltB)
    if typeof(result) == FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}
        return _RemoveZeros(result)
    end

    # Converts to default basis and multiplies.
    DefA = DefaultBasis(eltA.Parent.FreeModule)
    DefB = DefaultBasis(eltB.Parent.FreeModule)
    product = _Multiply(DefA, ToBasis(DefA, eltA.Parent, eltA), DefB, ToBasis(DefB, eltB.Parent, eltB))
    typeof(product) == FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType} || throw(error(string("Multiplication not defined between ", eltA.Parent.FreeModule, " and ", eltB.Parent.FreeModule)))

    # Changes back to either the A or the B basis. Prefers the left one.
    if product.Parent.FreeModule == eltA.Parent.FreeModule
        return _RemoveZeros(ToBasis(eltA.Parent, product.Parent, product))
    elseif product.Parent.FreeModule == eltB.Parent.FreeModule
        return _RemoveZeros(ToBasis(eltB.Parent, product.Parent, product))
    end
    throw(error(string("Multiplication result in module ", product.Parent.FreeModule, " was incompatible with either of ", eltA.Parent.FreeModule, " or ", eltB.Parent.FreeModule)))
end

# Fallback implementation for multiplication.
function _Multiply(A::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, eltA::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, B::FreeModuleBasis{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, eltB::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    return false
end

# Exponantiation.
function ^(elt::FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}, n::Int) where {BaseRingType, BaseRingElemType, IndexSetType, IndexType}
    n >= 0 || throw(ArgumentError(string("The exponent ", n, " must be nonnegative.")))
    if n == 0
        W = elt.Parent.FreeModule.IndexSet
        terms = Dict{IndexType, BaseRingElemType}()
        terms[one(W)] = one(elt.Parent.FreeModule.BaseRing)
        return FreeModuleElement{BaseRingType, BaseRingElemType, IndexSetType, IndexType}(elt.Parent, terms)
    end
    result = elt
    for k = 1:n-1
        result = result * elt
    end
    return result
end

"""
    HeckeAlgebra
From here on the base ring will be the Laurent polynomial ring over the integers and the index set will be a Coxeter group.
Reference: https://github.com/joelgibson/IHecke
"""

"""
    Coxeter groups
Additional functions for Coxeter groups.
"""
# Show a Coxeter element.
function Base.show(w::CoxElt)
    if Base.isone(w)
        print("[id]")
    else
        print(lexword(w))
    end
end

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
    HeckeAlgebra <: FreeMod{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}
A **HeckeAlgebra** is a free module over the Laurent polynomial ring over the integers with basis elements indexed by a Coxeter group.
"""
struct HeckeAlgebra <: FreeMod{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}
    BaseRing::LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing} # Commutative Ring this module is over. In case of the Hecke algebra it's the Laurent polynomial ring over the integers.
    IndexSet::CoxGrp # Coxeter group of finitely presented type.
end

# Function for creating a new Hecke algebra.
function HeckeAlgebra(W::CoxGrp)
    A, v = LaurentPolynomialRing(ZZ, "v")
    return HeckeAlgebra(A, W)
end

"""
    AntisphericalModule <: FreeMod{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}
The **AntisphericalModule** is a module which is naturally associated to the Hecke algebra and a standard parabolic subgroup.
"""
struct AntisphericalModule <: FreeMod{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}
    BaseRing::LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing} # Commutative Ring this module is over. In case of the antispherical module it's the Laurent polynomial ring over the integers.
    IndexSet::CoxGrp # Coxeter group of finitely presented type.
    Para::Array{Int, 1} # Parabolic subset of the Coxeter group.
end

# Function for creating a new antispherical module.
function AntisphericalModule(W::CoxGrp, I::Array{Int, 1})
    A, v = LaurentPolynomialRing(ZZ, "v")
    return AntisphericalModule(A, W, I)
end

"""
    SphericalModule <: FreeMod{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}
The **SphericalModule** is a module which is naturally associated to the Hecke algebra and a standard parabolic subgroup.
"""
struct SphericalModule <: FreeMod{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}
    BaseRing::LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing} # Commutative Ring this module is over. In case of the spherical module it's the Laurent polynomial ring over the integers.
    IndexSet::CoxGrp # Coxeter group of finitely presented type.
    Para::Array{Int, 1} # Parabolic subset of the Coxeter group.
end

# Function for creating a new spherical module.
function SphericalModule(W::CoxGrp, I::Array{Int, 1})
    A, v = LaurentPolynomialRing(ZZ, "v")
    return SphericalModule(A, W, I)
end

# Creates the basis element indexed by the nth element of the generating set S of the Coxeter group of A.
function getindex(A::FreeModuleBasis{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}, n::Int)
    n > 0 || throw(ArgumentError("n > 0 required"))
    W = A.FreeModule.IndexSet
    return A[CoxEltMat(W, W.simpRefls[n], W.simpRefls[n])]
end

# Creates the basis element indexed by an Array of integers.
function getindex(A::FreeModuleBasis{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}, L::Array{Int, 1})
    W = A.FreeModule.IndexSet
    M = identity_matrix(base_ring(W.cartanMat), rank(W))
    for i in L
        M = M * W.simpRefls[i]
    end
    return A[CoxEltMat(W, M, M^-1)]
end

# Needed for bar involution. Returns f(v^-1).
function twist(f::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}})
    A, v = LaurentPolynomialRing(ZZ, "v")
    return evaluate(change_base_ring(A, f), v^-1) # change_base_ring needed to get a datatype that works with evaluate.
end

# Perform the bar involution on elt, returning the result in the same basis as elt.
function Bar(elt::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt})
    # First check if there is a bar involution defined on this basis.
    result = _Bar(elt.Parent, elt)
    if typeof(result) == FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}
        return result
    end

    # Convert to canonical basis, perform Bar, convert back.
    K = KazhdanLusztigBasis(elt.Parent.FreeModule)
    result = ToBasis(elt.Parent, K, Bar(ToBasis(K, elt.Parent, elt)))
    typeof(result) == FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt} || throw(error(string("Bar involution not defined on", K)))
    return result
end

# Fallback implementation of Bar involution.
function _Bar(A::FreeModuleBasis{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}, eltA::FreeModuleElement)
    return false
end

"""
    BasisHeckeStd <: FreeModuleBasis{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}
An abstract type, used for the standard basis of an object of type FreeMod{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}.
"""
abstract type BasisHeckeStd <: FreeModuleBasis{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}
end

# Default basis of a FreeMod{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}
# is the standard basis.
DefaultBasis(FMod::FreeMod{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}) = StandardBasis(FMod)

# (M, eltM) is an element inside a right module over the Hecke algebra (either the algebra itself,
# or a parabolic module). Returns the right action of H[s] on eltM.
function _RightMultStdGen(M::BasisHeckeStd, eltM::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}, s::CoxElt)
    W = M.FreeModule.IndexSet
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
    return _FreeModuleElementValidate(M, _RemoveZeros(FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}(M, terms)))
end

# (M, eltM) is an element inside a right module over the Hecke algebra (either the algebra itself,
# or a parabolic module). This function returns eltM * H[terms], where terms is interpreted as a
# linear combination of standard basis elements of the Hecke algebra.
function _RightAction(M::BasisHeckeStd, eltM::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}, terms::Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}})
    acc = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
    W = M.FreeModule.IndexSet
    for w in keys(terms)
        piece = eltM
        for i in lexword(w)
            s = W.simpRefls[i]
            piece = _RightMultStdGen(M, piece, CoxEltMat(W, s, s))
        end
        _AddScaled(acc, piece.Terms, terms[w])
    end
    return _FreeModuleElementValidate(M, _RemoveZeros(FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}(M, acc)))
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
function _Bar(H::BasisHeckeStd, elt::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt})
    H == elt.Parent || throw(ArgumentError("H == elt.Parent required"))
    R = H.FreeModule.BaseRing
    terms = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
    for w in keys(elt.Terms)
        _AddScaled(terms, _BarInvolutionStd(H, w).Terms, twist(elt.Terms[w]))
    end
    return _FreeModuleElementValidate(H, _RemoveZeros(FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}(H, terms)))
end

"""
    HeckeAlgebraStdBasis <: BasisHeckeStd
A type, used for the standard basis of the Hecke algebra.
"""
struct HeckeAlgebraStdBasis <: BasisHeckeStd
    FreeModule::HeckeAlgebra # The Hecke algebra this basis belongs to.
    BasisSymbol::Union{AbstractString, Char, Symbol}  # A short string naming the basis. In case of the standard basis it's "H".
    BarCache::Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}
    # A dictionary, used as a cache for the bar involution on basis elements.
    Para::Array{Int, 1} # Parabolic subset (equals Array{Int, 1}() for the Hecke algebra).
    Eig::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}
    # Irrelevant for the full Hecke algebra , -v or v^-1 for the antispherical/spherical modules.
end

# Creates the standard basis of the Hecke algebra.
StandardBasis(HAlg::HeckeAlgebra) = HeckeAlgebraStdBasis(HAlg, "H", Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}(), Array{Int, 1}(), zero(HAlg.BaseRing))

# Multiplication inside the standard basis of the Hecke algebra.
function _Multiply(H1::HeckeAlgebraStdBasis, elt1::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}, H2::HeckeAlgebraStdBasis, elt2::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt})
    return _RightAction(H1, elt1, elt2.Terms)
end

"""
    ASModuleStdBasis <: BasisHeckeStd
A type, used for the standard basis of the antispherical module.
"""
struct ASModuleStdBasis <: BasisHeckeStd
    FreeModule::AntisphericalModule # The antispherical module this basis belongs to.
    BasisSymbol::Union{AbstractString, Char, Symbol}  # A short string naming the basis. In case of the standard basis it's "aH".
    BarCache::Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}
    # A dictionary, used as a cache for the bar involution on basis elements.
    Para::Array{Int, 1} # Parabolic subset (equals Array{Int, 1}() for the Hecke algebra).
    Eig::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}
    # Irrelevant for the full Hecke algebra , -v or v^-1 for the antispherical/spherical modules.
end

# Creates the standard basis of the antispherical module.
StandardBasis(ASMod::AntisphericalModule) = ASModuleStdBasis(ASMod, "aH", Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}(), ASMod.Para, -gen(ASMod.BaseRing))

# Only allow I-minimal elements.
function _FreeModuleElementValidate(aH::ASModuleStdBasis, elt::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt})
    I = aH.Para
    for w in keys(elt.Terms)
        IsMinimal(I, w) || throw(error(string(w, " is not minimal with respect to ", aH.FreeModule.Para)))
    end
    return elt
end

# Multiplication inside the standard basis of the antispherical module.
function _Multiply(M::ASModuleStdBasis, eltM::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}, H::HeckeAlgebraStdBasis, eltH::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt})
    return _RightAction(M, eltM, eltH.Terms)
end

"""
    SModuleStdBasis <: BasisHeckeStd
A type, used for the standard basis of the spherical module.
"""
struct SModuleStdBasis <: BasisHeckeStd
    FreeModule::SphericalModule # The spherical module this basis belongs to.
    BasisSymbol::Union{AbstractString, Char, Symbol}  # A short string naming the basis. In case of the standard basis it's "sH".
    BarCache::Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}
    # A dictionary, used as a cache for the bar involution on basis elements.
    Para::Array{Int, 1} # Parabolic subset (equals Array{Int, 1}() for the Hecke algebra).
    Eig::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}
    # Irrelevant for the full Hecke algebra , -v or v^-1 for the antispherical/spherical modules.
end

# Creates the standard basis of the spherical module.
StandardBasis(SMod::SphericalModule) = SModuleStdBasis(SMod, "sH", Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}(), SMod.Para, gen(SMod.BaseRing)^-1)

# Only allow I-minimal elements.
function _FreeModuleElementValidate(sH::SModuleStdBasis, elt::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt})
    I = sH.Para
    for w in keys(elt.Terms)
        IsMinimal(I, w) || throw(error(string(w, " is not minimal with respect to sH.FreeModule.Para")))
    end
    return elt
end

# Multiplication inside the standard basis of the spherical module.
function _Multiply(M::SModuleStdBasis, eltM::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}, H::HeckeAlgebraStdBasis, eltH::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt})
    return _RightAction(M, eltM, eltH.Terms)
end

"""
    BasisHeckeKL <: BasisHecke
An abstract type, used for the Kazhdan-Lusztig basis of an object of type FreeModuleCox.
"""
abstract type BasisHeckeKL <: FreeModuleBasis{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}
end

# Assume that elt is an element written in terms of the standard basis, and right-multiply
# by the element B[s]. I is the parabolic quotient (Array{Int, 1}() if we are in the Hecke algebra), and eig
# is -v for antispherical, v^-1 for spherical, and irrelevant if in the Hecke algebra.
function _RightMultKLGen(elt::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}, s::CoxElt, I::Array{Int, 1}, eig::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}})
    W = elt.Parent.FreeModule.IndexSet
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
    return _FreeModuleElementValidate(P, _RemoveZeros(FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}(P, terms)))
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
    W = H.FreeModule.IndexSet
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
    W = H.FreeModule.IndexSet
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
    result = _FreeModuleElementValidate(K, _RemoveZeros(FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}(K, terms)))
    K.StdInKLCache[w] = result
    return result
end

# The bar involution of elt, fixing each basis element and twisting scalars by v -> v^-1.
function _Bar(K::BasisHeckeKL, elt::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt})
    K == elt.Parent || throw(ArgumentError("K == elt.Parent required"))
    terms = Dict{CoxElt, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}}()
    for w in keys(elt.Terms)
        terms[w] = twist(elt.Terms[w])
    end
    return _FreeModuleElementValidate(K, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}(K, terms))
end

"""
    HeckeAlgebraKLBasis <: BasisHeckeKL
A type, used for the Kazhdan-Lusztig basis of the Hecke algebra.
"""
struct HeckeAlgebraKLBasis <: BasisHeckeKL
    FreeModule::HeckeAlgebra # The Hecke algebra this basis belongs to.
    BasisSymbol::Union{AbstractString, Char, Symbol} # A short string naming the basis. In case of the Kazhdan-Lusztig basis it's "K".
    KLInStdCache::Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}
    # A dictionary, used as a cache for the basis change from Kazhdan-Lusztig basis to standard basis.
    StdInKLCache::Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}
    # A dictionary, used as a cache for the basis change from standard basis to Kazhdan-Lusztig basis.
    MuCache::Dict{CoxElt, Dict{CoxElt, Nemo.fmpz}} # A dictionary, used as a cache for the coefficient mu.
    Para::Array{Int, 1} # Parabolic subset (equals Array{Int, 1}() for the Hecke algebra).
    Eig::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}
    # Irrelevant for the full Hecke algebra , -v or v^-1 for the antispherical/spherical modules.
end

# Creates the Kazhdan-Lusztig basis of the Hecke algebra.
KazhdanLusztigBasis(HAlg::HeckeAlgebra) = HeckeAlgebraKLBasis(HAlg, "K", Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}(), Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}(), Dict{CoxElt, Dict{CoxElt, Nemo.fmpz}}(), Array{Int, 1}(), zero(HAlg.BaseRing))

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
function _Multiply(K1::HeckeAlgebraKLBasis, eltA::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}, K2::HeckeAlgebraKLBasis, eltB::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt})
    K1 == K2 || throw(ArgumentError("K1 == K2 required"))
    H = StandardBasis(K1.FreeModule)
    W = K1.FreeModule.IndexSet
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
                return _FreeModuleElementValidate(K1, _RemoveZeros(FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}(K1, terms)))
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
                return _FreeModuleElementValidate(K1, _RemoveZeros(FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}(K1, terms)))
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
    KLInStdCache::Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}
    # A dictionary, used as a cache for the basis change from Kazhdan-Lusztig basis to standard basis.
    StdInKLCache::Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}
    # A dictionary, used as a cache for the basis change from standard basis to Kazhdan-Lusztig basis.
    MuCache::Dict{CoxElt, Dict{CoxElt, Nemo.fmpz}} # A dictionary, used as a cache for the coefficient mu.
    Para::Array{Int, 1} # Parabolic subset (equals Array{Int, 1}() for the Hecke algebra).
    Eig::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}
    # Irrelevant for the full Hecke algebra , -v or v^-1 for the antispherical/spherical modules.
end

# Creates the Kazhdan-Lusztig basis of the antispherical module.
KazhdanLusztigBasis(ASMod::AntisphericalModule) = ASModuleKLBasis(ASMod, "aK", Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}(), Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}(), Dict{CoxElt, Dict{CoxElt, Nemo.fmpz}}(), ASMod.Para, -gen(ASMod.BaseRing))

# Only allow I-minimal elements.
function _FreeModuleElementValidate(aK::ASModuleKLBasis, elt::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt})
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
    KLInStdCache::Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}
    # A dictionary, used as a cache for the basis change from Kazhdan-Lusztig basis to standard basis.
    StdInKLCache::Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}
    # A dictionary, used as a cache for the basis change from standard basis to Kazhdan-Lusztig basis.
    MuCache::Dict{CoxElt, Dict{CoxElt, Nemo.fmpz}} # A dictionary, used as a cache for the coefficient mu.
    Para::Array{Int, 1} # Parabolic subset (equals Array{Int, 1}() for the Hecke algebra).
    Eig::LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}
    # Irrelevant for the full Hecke algebra , -v or v^-1 for the antispherical/spherical modules.
end

# Creates the Kazhdan-Lusztig basis of the spherical module.
KazhdanLusztigBasis(SMod::SphericalModule) = SModuleKLBasis(SMod, "sK", Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}(), Dict{CoxElt, FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt}}(), Dict{CoxElt, Dict{CoxElt, Nemo.fmpz}}(), SMod.Para, gen(SMod.BaseRing)^-1)

# Only allow I-minimal elements.
function _FreeModuleElementValidate(sK::SModuleKLBasis, elt::FreeModuleElement{LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}, LaurentPolyWrap{Nemo.fmpz, Nemo.fmpz_poly, LaurentPolyWrapRing{Nemo.fmpz, Nemo.FmpzPolyRing}}, CoxGrp, CoxElt})
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
(K[1] + K[2])*K[3]
K[1]*(K[2] + K[3])
H[3]*(H[1] + v*H[[1, 2, 3]])
2*H[3]*K[2]^3 + (v^2 + v^-1)*K[1]
Bar(K[1])
Bar(H[3])
Bar(2*H[3]*K[2]^3 + (v^2 + v^-1)*K[1])
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

struct ABCModule <: FreeMod{Nemo.FlintIntegerRing, Nemo.fmpz, Array{String, 1}, String}
    BaseRing::Nemo.FlintIntegerRing
    IndexSet::Array{String, 1}
end

ABCModule(A::Array{String, 1}) = ABCModule(ZZ, A)

struct ABCModuleBasis <: FreeModuleBasis{Nemo.FlintIntegerRing, Nemo.fmpz, Array{String, 1}, String}
    FreeModule::ABCModule
    BasisSymbol::Union{AbstractString, Char, Symbol}
end

ABCModuleBasis(ABCMod::ABCModule) = ABCModuleBasis(ABCMod, "X")

function Base.show(a::String)
    print("[", a, "]")
end

ABCMod = ABCModule(["a", "b", "c"])
X = ABCModuleBasis(ABCMod)
X["a"] + X["b"]
3*X["a"]

function _Multiply(A::ABCModuleBasis, eltA::FreeModuleElement{Nemo.FlintIntegerRing, Nemo.fmpz, Array{String, 1}, String}, B::ABCModuleBasis, eltB::FreeModuleElement{Nemo.FlintIntegerRing, Nemo.fmpz, Array{String, 1}, String})
    terms = Dict{String, Nemo.fmpz}()
    terms["a"] = terms["b"] = terms["c"] = 0
    for w in keys(eltA.Terms)
        scalarA = eltA.Terms[w]
        for s in keys(eltB.Terms)
            result = scalarA * eltB.Terms[s]
            if (w == "a" && s == "a") || (w == "b" && s == "c") || (w == "c" && s == "b")
                terms["a"] = terms["a"] + result
            elseif (w == "a" && s == "b") || (w == "b" && s == "a") || (w == "c" && s == "c")
                terms["b"] = terms["b"] + result
            else
                terms["c"] = terms["c"] + result
            end
        end
    end
    return FreeModuleElement{Nemo.FlintIntegerRing, Nemo.fmpz, Array{String, 1}, String}(A, terms)
end

one(W::Array{String, 1}) = "a"

X["a"]*X["b"]
2*X["a"]*(X["b"] + 3*X["c"])*X["c"]
(X["a"] + X["b"])^10

struct ABCModuleBasis2 <: FreeModuleBasis{Nemo.FlintIntegerRing, Nemo.fmpz, Array{String, 1}, String}
    FreeModule::ABCModule
    BasisSymbol::Union{AbstractString, Char, Symbol}
end

ABCModuleBasis2(ABCMod::ABCModule) = ABCModuleBasis2(ABCMod, "Y")

DefaultBasis(ABCMod::ABCModule) = ABCModuleBasis(ABCMod)

function _ToBasis(A::ABCModuleBasis, B::ABCModuleBasis2, w::String)
    if w == "a"
        return A["a"]
    elseif w == "b"
        return A["a"] + A["b"]
    else
        return A["a"] + A["b"] + A["c"]
    end
end

function _ToBasis(B::ABCModuleBasis2, A::ABCModuleBasis, w::String)
    if w == "a"
        return B["a"]
    elseif w == "b"
        return B["b"] - B["a"]
    else
        return B["c"] - B["a"]
    end
end

Y = ABCModuleBasis2(ABCMod)
X["a"]*Y["b"]
Y:2*X["a"]*(X["b"] + 3*X["c"])*X["c"]
