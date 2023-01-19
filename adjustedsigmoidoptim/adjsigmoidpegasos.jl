using MLJ, Tables, LinearAlgebra, Plots

function smooth_pegasos(X, Y, λ, T)
    m, n = size(X)
    w = zeros(n)
    S = 1:m |> collect
    for t in 1:T
        i = rand(S)
        η = 1 / (λ * t)
        yi = Y[i]
        xi = X[i, :]
        w = (1 - 1/t) .* w .+ η * yi * (yi - tanh(dot(w, xi))) ^ 2 .* xi
        w = min(1, (1/sqrt(λ)) / norm(w)) .* w
    end
    return w
end

N = 1000
D = make_blobs(1000, 2000, centers = 2)
X = D[1] |> Tables.matrix |> eachrow .|> collect
Y = (D[2] .== 1 .|> Float64) .* 2 .- 1
Y

Xv = D[1] |> Tables.matrix
scatter(Xv[:, 1], Xv[:, 2], group = Y, legend = false)

w = smooth_pegasos(Xv, Y, 10, 100)
pred = Xv * w .|> tanh .|> sign
l = @layout [a;b;c]

begin
    p1 = scatter(Xv[:, 1], Xv[:, 2], group = Y, legend = false)
    p2 = scatter(Xv[:, 1], Xv[:, 2], group = pred, legend = false)
    wrong = (x -> x ? "wrong" : "right").(pred .!= Y)
    p3 = scatter(Xv[:, 1], Xv[:, 2], group = wrong,legend = false)
    plot(p1, p2, p3, layout = l)
end

w |> sort |> reverse |> plot