using Flux
using Plots
using Zygote
using Images

rbf(x, μ, σ) = exp.(-(x .- μ) .^ 2 ./ (2σ^2))


X = 0:.1:12 |> collect
Y = 3 .* X .^ 2 .+ (rand(length(X)) .* 10)

scatter(X, Y, legend = false)

K = [rbf(x, μ, .2) for x in X, μ in X]

rbfx(x) = μ -> rbf(x, μ, .2)

rbfx(X[1]).(X)

Ks = K' |> eachrow .|> collect
IL = X |> length
begin
begin
    d1 = Dense(IL, 10)
    d12 = Dense(IL, 10)
    d13 = Dense(IL, 10)
    d2 = Dense(30, 1)
    drop = Dropout(.75)
    IL2 = 2
    md1 = Dense(1, IL2, relu)
    md2 = Dense(IL2, 1)
    decider = Dense(2, 1)
end

m = Chain(d1, d2, d13, d2, drop, md1, md2, decider)

function model1(x)
    x1 = d1(x)
    x12 = d12(x)
    x13 = d13(x)
    d2i = [x1; x12; x13] |> drop
    x2 = d2(d2i)
    return x2
end

function model2(x)
    x1 = md1([x])
    x2 = md2(x1)
    return x2
end

function model(x)
    i1 = rbfx(x).(X) |> model1
    # i2 = model2(x)
    # decider([i1;i2])
    return i1
end


params = Flux.params(m)
opt = ADAM(.1)

sqr(x) = x^2

losss(X, Y) = sum(abs2, reduce(vcat, model.(X)) .- Y)
losses = []
for _ in 1:100
    train_loss, back = Zygote.pullback(params) do
        model1.(Ks) 
    end
    push!(losses, train_loss)
    gs = back(train_loss)
    Flux.Optimise.update!(opt, params, gs)
end

scatter(X, Y)

iX = -5:.1:15 
iY = reduce(vcat, model.(iX))
plot!(-5:.1:15, iY, label="model", legend = :outertopright)
end
# plot(1:length(losses), losses, label="loss")

