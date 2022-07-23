abstract type AbstractGraphDomain end

struct NullDomain <: AbstractGraphDomain
end

domain(::NullDomain) = nothing
positional_feature(::NullDomain) = nothing
has_positional_feature(::NullDomain) = false
pf_dims_repr(::NullDomain) = 0
check_num_nodes(graph_nv::Real, ::NullDomain) = nothing


struct NodeDomain{T} <: AbstractGraphDomain
    domain::T
end

NodeDomain(::Nothing) = NullDomain()
NodeDomain(d::AbstractGraphDomain) = d

@functor NodeDomain

domain(d::NodeDomain) = d.domain
positional_feature(d::NodeDomain) = d.domain
has_positional_feature(::NodeDomain) = true
pf_dims_repr(d::NodeDomain) = size(d.domain, 1)
check_num_nodes(graph_nv::Real, d::NodeDomain) = check_num_nodes(graph_nv, size(d.domain, 2))
