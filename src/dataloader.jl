function MLUtils.numobs(fg::FeaturedGraph)
    obs_size = 0
    if has_node_feature(fg)
        nf_obs_size = numobs(node_feature(fg))
        obs_size = check_obs_size(obs_size, nf_obs_size, "node features")
    end
    if has_edge_feature(fg)
        ef_obs_size = numobs(edge_feature(fg))
        obs_size = check_obs_size(obs_size, ef_obs_size, "edge features")
    end
    if has_global_feature(fg)
        gf_obs_size = numobs(global_feature(fg))
        obs_size = check_obs_size(obs_size, gf_obs_size, "global features")
    end
    if has_positional_feature(fg)
        pf_obs_size = numobs(positional_feature(fg))
        obs_size = check_obs_size(obs_size, pf_obs_size, "positional features")
    end
    return obs_size
end

function check_obs_size(obs_size, feat_obs_size, feat::String)
    if obs_size != 0
        msg = "inconsistent number of observation between $feat ($feat_obs_size) and others ($obs_size)"
        @assert obs_size == feat_obs_size msg
    end
    return feat_obs_size
end

MLUtils.getobs(fg::FeaturedGraph, idx) =
    FeaturedGraph(fg, nf=getobs(node_feature(fg), idx), ef=getobs(edge_feature(fg), idx),
                  gf=getobs(global_feature(fg), idx), pf=getobs(positional_feature(fg), idx))
