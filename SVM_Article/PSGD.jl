using MLJ, LinearAlgebra, Tables, Plots

J(w, λ, x, y) = λ/2 * norm(w) ^ 2 + max(0, 1 - y * dot(w, x))
Indicator(x) = x |> Float64
dJ(w, λ, x, y) = λ .* w .- (Indicator(y * dot(w, x) < 1) * y) .* x 

function pegasos(X, Y, λ, T)
    m, n = size(X)
    w = zeros(n)
    S = 1:m |> collect
    for t in 1:T
        i = rand(S)
        η = 1 / (λ * t)
        yi = Y[i]
        xi = X[i, :]
        w = (1 - 1/t) .* w .+ (η * Indicator(yi * dot(w, xi) < 1) * yi) .* xi
        w = min(1, (1/sqrt(λ)) / norm(w)) .* w
    end
    return w
end

Mblobs = MLJ.make_moons(1000)
X = Mblobs[1] |> Tables.matrix
Y = Vector{Float64}(Mblobs[2])
D = zip(eachrow(X) .|> collect, Y) |> collect
J(w, λ) = d -> J(w, λ, d[1], d[2])
rbf(x, y) = exp(-norm(x - y) ^ 2 / 2)
K = [rbf(X[i,:], X[j,:]) for i in 1:size(X)[1], j in 1:size(X)[1]]
Dk = zip(eachrow(K), Y) |> collect
λ = 10e-6

begin
    w = pegasos(X, Y, λ, 1000)
    pred = J(w, λ).(D) .< 1
    wrong = (Y .- pred) .== 0
    print(w)
    scatter(X[:, 1], X[:, 2], group = wrong)
end


begin
    w = pegasos(K, Y, λ, 1000)
    pred = J(w, λ).(Dk) .< 1
    wrong = (Y .- pred) .== 0
    print(w)
    scatter(X[:, 1], X[:, 2], group = wrong)
end

histogram(w, bins = 100)