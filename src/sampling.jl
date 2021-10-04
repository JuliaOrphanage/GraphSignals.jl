abstract type AbstractGraphPreprocessor <: AbstractFeaturedGraph end
abstract type AbstractGraphSampler <: AbstractGraphPreprocessor end
abstract type AbstractDeterministicGraphSampler <: AbstractGraphSampler end
abstract type AbstractProbabilisticGraphSampler <: AbstractGraphSampler end

struct GraphSampler <: AbstractProbabilisticGraphSampler
    fg::AbstractFeaturedGraph
    prob::Real  # the probability to sample a feature
end

function graph(gs::GraphSampler)
    A = adjacency_matrix(gs.fg)
    A .* (rand(size(A)...) .< gs.prob)
end

# node_feature(gs::GraphSampler)
# edge_feature(gs::GraphSampler)
global_feature(gs::GraphSampler) = global_feature(gs.fg)
