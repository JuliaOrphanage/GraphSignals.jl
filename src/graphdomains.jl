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

# Bug caused by dispatching rrule for NodeDomain to the following function:
# https://github.com/FluxML/Flux.jl/blob/c4837f74a767f9ed9a0919626792460b93ee7995/src/functor.jl#L118
ChainRulesCore.rrule(::Type{NodeDomain}, x::CUDA.CuArray) =
    NodeDomain(x), d -> (NoTangent(), d.domain)

Adapt.adapt_structure(to, d::NodeDomain) = NodeDomain(Adapt.adapt(to, d.domain))

domain(d::NodeDomain) = d.domain
positional_feature(d::NodeDomain) = d.domain
has_positional_feature(::NodeDomain) = true
pf_dims_repr(d::NodeDomain) = size(d.domain, 1)
check_num_nodes(graph_nv::Real, d::NodeDomain) = check_num_nodes(graph_nv, size(d.domain, 2))
