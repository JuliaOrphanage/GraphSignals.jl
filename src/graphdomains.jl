abstract type AbstractGraphDomain end

struct NullDomain <: AbstractGraphDomain
end

domain(::NullDomain) = nothing
positional_feature(::NullDomain) = nothing
has_positional_feature(::NullDomain) = false


struct NodeDomain{T} <: AbstractGraphDomain
    domain::T
end

@functor NodeDomain

domain(d::NodeDomain) = d.domain
positional_feature(d::NodeDomain) = d.domain
has_positional_feature(::NodeDomain) = true

