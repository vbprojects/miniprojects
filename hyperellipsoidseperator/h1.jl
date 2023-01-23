using Zygote, MLJ, Plots, Images

X, y = make_moons(1000, noise=0.1, as_table = false)

Y = 2 * y .- 1

clf(w, b, x) = begin
    sum(
        x[i]^2 / b[i]^2 - 2 * x[i] * w[i] / b[i]^2 + w[i]^2 / b[i]^2 for i in eachindex(x)
    )
end

function loss(w, b, X, y)
    sum(max(0, 1 - clf(w, b, X[i, :]) * y[i]) for i in eachindex(y))
end

function grad(w, b, X, y)
    Zygote.gradient((w, b) -> loss(w, b, X, y), w, b)
end

begin
    losses = []
    w = rand(2)
    b = rand(2)
    for _ in 1:1000
        push!(losses, loss(w, b, X, Y))
        g = grad(w, b, X, Y)
        w -= 0.001 * g[1]
        b -= 0.001 * g[2]
    end
end

plot(losses[10:end])

yp = [clf(w, b, X[i, :]) for i in eachindex(y)]

scatter(X[:,1], X[:, 2], zcolor = yp)

cls = (x, y) -> clf(w, b, [x y])

clf(w, b, [1 2])

[cls(x, y) for x in -10:0.1:10, y in -10:0.1:10] .|> Gray