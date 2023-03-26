using Plots, Zygote

module tem
    mutable struct particle
        vel :: Array{Float64,1}
        pos :: Array{Float64,1}
        depth :: Float64
    end
end
kern(x) = [xi^i for i in 0:3 for xi in x]
particle = tem.particle
c = collect
X = -1:.01:1 |> c
Y = [Ϛ(X[i], [.5;randn(3)], [.5;randn(3)], kern) for i in axes(X, 1)]
Ϛ(x, α, β, ϕ) = β'*sin.(α .* ϕ(x))

plot(X, Y)
loss(X, Y, α, β, ϕ) = sum((Ϛ(X[i, :], α, β, ϕ) - Y[i])^2 for i in axes(X, 1))
kern(X[1])
α = randn(1)
β = randn(1)

kern(x) = x

loss(X, Y, α, β, kern)

plotly()

surface(-10:.1:10, -10:.1:10, (x, y) -> loss(X, Y, [y;1;0;0], [x;1;0;0], kern))

# codex auto generated this, WHERE IS THIS FROM?
function swarm_tactics(X, Y, ϕ, α, β, ϵ, ρ, ω, max_iter)
    N = size(X, 1)
    M = size(X, 2)
    P = [particle(randn(M), randn(M), Inf) for i in 1:N]
    G = particle(randn(M), randn(M), Inf)
    for i in 1:max_iter
        for j in 1:N
            P[j].vel = ω * P[j].vel + ϵ * rand() * (G.pos - P[j].pos) + ρ * rand() * (P[rand(1:N)].pos - P[j].pos)
            P[j].pos += P[j].vel
            P[j].depth = loss(X, Y, α, β, ϕ)
            if P[j].depth < G.depth
                G = P[j]
            end
        end
    end
    return G
end



