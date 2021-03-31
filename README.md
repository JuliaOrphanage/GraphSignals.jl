# GraphSignals.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://yuehhua.github.io/GraphSignals.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://yuehhua.github.io/GraphSignals.jl/dev)
[![Build Status](https://travis-ci.org/yuehhua/GraphSignals.jl.svg?branch=master)](https://travis-ci.org/yuehhua/GraphSignals.jl)
[![pipeline status](https://gitlab.com/JuliaGPU/GraphSignals.jl/badges/master/pipeline.svg)](https://gitlab.com/JuliaGPU/GraphSignals.jl/commits/master)
[![coverage report](https://gitlab.com/JuliaGPU/GraphSignals.jl/badges/master/coverage.svg)](https://gitlab.com/JuliaGPU/GraphSignals.jl/commits/master)

A generic graph representation for combining graph signals (or features) and graph topology (or graph structure). It supports the graph structure defined in JuliaGraphs packages (i.e. LightGraphs and SimpleWeightedGraphs) and compatible with APIs in JuliaGraphs packages. Graph signals are usually features, including node feautres, edge features and graph features. Features are contained in arrays and CuArrays are supported via CUDA.jl.

## APIs

### Graph-related APIs

* `graph`
* `node_feature`
* `edge_feature`
* `global_feature`
* `mask`
* `has_graph`
* `has_node_feature`
* `has_edge_feature`
* `has_global_feature`
* `nv`
* `ne`
* `adjacency_list`
* `is_directed`
* `fetch_graph`

### Linear algebraic APIs

* `adjacency_matrix`
* `degrees`
* `degree_matrix`
* `inv_sqrt_degree_matrix`
* `laplacian_matrix`, `laplacian_matrix!`
* `normalized_laplacian`, `normalized_laplacian!`
* `scaled_laplacian`, `scaled_laplacian!`
