using Plots, Flux, Images, LinearAlgebra, Zygote, CUDA, StatsBase


X = -3:.1:10 |> collect

function randindinit(n,X,Y)
    distv(Y, y) = broadcast.(norm, Y, y)
    xs = zeros(Int, n)
    xs[1] = rand(1:length(X))
    xd1 = distv(X, X[xs[1]])
    xd = Dict(1 => xd1 .^ 2, 0 => xd1 .^ 2)
    for i in 2:n
        xs[i] = sample(1:length(X), Weights(xd[i-1] .+ xd[i-2]))
        xd[i] = distv(Y, Y[xs[i]]) .^ 2
    end
    return xs
end

cas(x) = cos(x) + sin(x)
f(x) = sinc(x) + cas(3 * x)
Y = f.(X) .+ rand(length(X)) .* .5
scatter(X, Y, legend = false)

rbf(x) = exp.(-1.0 .* x .^ 2 ./ 1)
sink(x) = sinc.(-1.0 .* x .^ 2 ./ 1)
cosk(x) = cos.(-1.0 .* x .^ 2 ./ 1)
N = 10

begin
    scatter(X, Y, legend = :outertopright, markersize = 2, xlabel = "X", ylabel = "Y", label = "Data")
    rbfinds = randindinit(N, X, Y)#sample(1:length(X), N, replace = false)
    K = reduce(hcat, rbf.(broadcast.(-, X, (X[rbfinds],))))'
    α = pinv(K)*Y 
    Ks = reduce(hcat, cosk.(broadcast.(-, X, (X[rbfinds],))))'
    αs = pinv(Ks)*Y 

    fullK = hcat(K, Ks, X)
    afull = pinv(fullK)*Y

    Xn = -3:.01:10 |> collect
    Xnα = reduce(hcat, rbf.(broadcast.(-, Xn, (X[rbfinds],))))'
    Xnαs = reduce(hcat, cosk.(broadcast.(-, Xn, (X[rbfinds],))))'
    XnfullK = hcat(Xnα, Xnαs, Xn)
    XnY = XnfullK * afull
    
    plot!(Xn, XnY, label = "full")
    scatter!(X[rbfinds], Y[rbfinds], label = "samples")
    plot!(Xn, f.(Xn), label = "true")
    # plot!(X, K*α, label = "rbf")
    # plot!(X, Ks*αs, label = "sink")
    # plot!(X, X*(pinv(X)*Y), label = "linear")
end
begin
    plot(afull, label = "full")
    plot!(α, label = "rbf")
    plot!(αs, label = "sink")
end

sumabs(x) = sum(abs2, x)

Y .- fullK*afull |> sumabs
Y .- K*α |> sumabs
Y .- Ks*αs |> sumabs
Y .- X*(pinv(X)*Y) |> sumabs 



randindinit(10, X, Y)
