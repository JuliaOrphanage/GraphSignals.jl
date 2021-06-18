# From FluxML/NNlib.jl/test/test_utils.jl
using ChainRulesTestUtils
using FiniteDifferences
using Zygote

function gradtest(f, xs...; atol=1e-6, rtol=1e-6, fkwargs=NamedTuple(),
                    check_rrule=false,
                    fdm=:central, 
                    check_broadcast=false,
                    skip=false, broken=false)
    # TODO: revamp when https://github.com/JuliaDiff/ChainRulesTestUtils.jl/pull/166
    # is merged
    if check_rrule
        test_rrule(f, xs...; fkwargs=fkwargs)
    end

    if check_broadcast
        length(fkwargs) > 0 && @warn("CHECK_BROADCAST: dropping keywords args")
        h = (xs...) -> sum(sin.(f.(xs...)))
    else
        h = (xs...) -> sum(sin.(f(xs...; fkwargs...)))
    end

    y_true = h(xs...)
    if fdm == :central
        fdm_obj = FiniteDifferences.central_fdm(5, 1)
    elseif fdm == :forward
        fdm_obj = FiniteDifferences.forward_fdm(5, 1)
    elseif fdm == :backward
        fdm_obj = FiniteDifferences.backward_fdm(5, 1)
    end
    # @show fdm fdm_obj

    gs_fd = FiniteDifferences.grad(fdm_obj, h, xs...)

    y_ad, pull = Zygote.pullback(h, xs...)
    gs_ad = pull(one(y_ad))

    @test y_true ≈ y_ad  atol=atol rtol=rtol
    for (g_ad, g_fd) in zip(gs_ad, gs_fd)
        if skip
            @test_skip g_ad ≈ g_fd   atol=atol rtol=rtol
        elseif broken
            @test_broken g_ad ≈ g_fd   atol=atol rtol=rtol
        else
            @test g_ad ≈ g_fd   atol=atol rtol=rtol
        end
    end
    return true
end