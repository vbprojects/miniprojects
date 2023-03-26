using Flux, Flux.Optimise
using Flux: onehotbatch, onecold, crossentropy, throttle
using MLUtils: mapobs, DataLoader, flatten
using Images: Gray
using CUDA: CuIterator, @cuprint
using ProgressMeter: @showprogress 
using Plots, StatsPlots, StatsBase
using Base.Iterators
using Zygote
using LinearAlgebra
using Plots

#download MNIST Dataset
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
using MLDatasets
train_x, train_y = MNIST.traindata(Float32)
test_x, test_y = MNIST.testdata(Float32)

toint(x) = floor(x) |> Int

layer_ws = [2, 1.5, 1.2, 1]
w1 = layer_ws[1:end-1]
w2 = layer_ws[2:end]
encoder = Chain([Dense(28^i |> toint, 28^j |> toint, relu) for (i, j) in zip(w1, w2)]...)
decoder = Chain([Dense(28^i |> toint, 28^j |> toint, relu) for (i, j) in reverse(zip(w2, w1) |> collect)]...)
model = Chain(encoder, decoder)

data = DataLoader((train_x |> flatten, train_x |> flatten), batchsize=128, shuffle=true)
loss = Flux.mse
opt = ADAM(0.01)
p = Flux.params(model)

# train
for epoch in 1:20
    println("Epoch: $epoch")
    Flux.train!(p, data, opt) do x, y
        z = encoder(x)
        loss(decoder(z), y)
    end
end

rs(x, y) = z -> reshape(z, (x, y))

encoder(train_x[:, rand(1:6000)]) |> decoder |> rs(28, 28) .|> Gray
hyperellipsoid(w, b, x, r) = ((x .- w) .^ 2)'*(b .^ -2) - r
sgdloss(w, b, x, y, r) = max(0, 1 - hyperellipsoid(w, b, x, r) * y)
sgdgrad(w, b, x, y, r) = Zygote.gradient((w, b, r) -> sgdloss(w, b, x, y, r), w, b, r)

function train(X, Y)
    # n = size(X, 1)
    w = mean(X[Y .== -1, :], dims = 1)[1, :]
    b = std(X[Y .== -1, :], dims = 1)[1, :] .+ 1e-5 # add a small constant to avoid division by zero
    r = 1
    # losses = []
    for _ in 1:10000
        # push!(losses, sum(sgdloss(w, b, X[i, :], Y[i], r) for i in rand(1:size(X, 1), 1000)))
        i = rand(1:size(X, 1))
        x = X[i, :]
        y = Y[i]
        g = sgdgrad(w, b, x, y, r)
        w -= 0.001 * g[1]
        b -= 0.001 * g[2]
        r -= 0.001 * g[3]
    end
    w, b, r
end

using StatsPlots



yn = [2 .* (train_y .== j) .- 1 for j in 1:9]


ftx = train_x |> flatten
tx = [encoder(ftx[:, i]) for i in 1:size(ftx, 2)]

TX = reduce(hcat, tx)'

FX = train_x |> flatten |> transpose

FX = TX

w, b, r = train(FX, y)

fe_f = [train(FX, yn[i]) for i in 1:9]

d_mu = [norm(fe_f[i][1] - fe_f[j][1]) for i in 1:9, j in 1:9]
d_b = [norm(fe_f[i][2] - fe_f[j][2]) for i in 1:9, j in 1:9]
d_r = [norm(fe_f[i][3] - fe_f[j][3]) for i in 1:9, j in 1:9]

function pred(w, b, r, X)
    [hyperellipsoid(w, b, X[i, :], r) + r for i in 1:size(X, 1)]
end

eng = [pred(w, b, r, FX) for (w, b, r) in fe_f]

fe = exp.(-1 .* reduce(hcat, eng) / 10)

function pegasos(X, Y, λ, T)
    Indicator(x) = x |> Float64
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
rbfe = exp.(-1 .* fe ./ .1)
function trial(i)
    # l = @layout [a;b]
    y = 2 .* (train_y .== i) .- 1
    m = pegasos(fe, y, 0.0001, 10000)
    # p1 = density([m'fe[i,:] for i in 1:1000], group = y[1:1000], bins = 100)
    m2 = pegasos(TX, y, 0.0005, 10000)
    # p2 = density([m2'TX[i,:] for i in 1:1000], group = y[1:1000], bins = 100)
    [(sign.(fe*m) .!= y) |> mean, (sign.(FX*m2) .!= y) |> mean, abs.(m) |> sum, abs.(m2) |> sum]
end

pegasos(TX, y, 0.0005, 10000)

comp = [[trial(i) for i in 1:9] for _ in 1:10]
cx = reduce(hcat, comp)
l = @layout [a b c; d e f; g h i]
gr(size=(800,800), dpi=800)
begin
    ps = []
    ns = []
    for i in 1:9
        probs = reduce(hcat, cx[i, :])
        density(probs[1, :], legend = false, color = :red, xlims = (0, 1))
        p = density!(probs[2, :], legend = false, color = :blue)
        push!(ps, p)
        density(probs[3, :], legend = false, color = :red)
        n = density!(probs[4, :], legend = false, color = :blue)
        push!(ns, n)
    end
    plot(ps..., layout = l)
end

cx
reduce(hcat, cx[1, :])
# scatter(fe[1:600, 1], fe[1:600, 4], group = train_y[1:600])