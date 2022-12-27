using Plots, Zygote, Random

cluster(x, y, z, n) = hcat(randn(n) .+ x, randn(n) .+ y, randn(n) .+ z)
function clusters(L, n)
    cs = [cluster(p[1], p[2], p[3], n) for p in L]
    ts = [ones(n) * i for i in eachindex(L)]
    X = vcat(cs...)
    Y = vcat(ts...)
    X, Y
end

Xd2, Y = clusters([(0, 3, 0), (0, 0, 0)], 100)
X = Xd2[:, 1:2]
Y .-= 1

plot(X[:, 1], X[:, 2], group = Y, seriestype = :scatter, legend = false)

s(x) = 1 / (1 + exp(-x))

dw(w, x, y) = abs(y - s(w' * x)) * (2 * y - 1) * x

cdw(w, x, y) = (y/s(w' * x) - (1 - y)/(1 - s(w' * x))) * x

tanhdw(w, x, y) = (y - tanh(w' * x))^2 *y * x

aX = hcat(X, ones(size(X)[1]))

sum(abs.(Y .- s.(aX * w)) .* (2 .* Y .- 1) .* aX, dims = 1)

# Surrogate batch gradient descent
sbSSE = []
sbENTROPY = []
w = zeros(3)
for _ in 1:100
    w .+= .05 * sum(abs.(Y .- s.(aX * w)) .* (2 .* Y .- 1) .* aX, dims = 1)'
    push!(sbSSE, sum(abs2, Y .- s.(aX * w)))
    push!(sbENTROPY, sum(-Y .* log.(s.(aX * w)) .- (1 .- Y) .* log.(1 .- s.(aX * w))))
end
plot(sbSSE, label = "SSE")
sbSSE[end]
w
# Surrogate perceptron gradient descent

sSSE = []
sENTROPY = []
w = zeros(3)
for _ in 1:100
    for i in shuffle(1:size(X)[1])
        w .+= .1 * dw(w, aX[i, :], Y[i])
    end
    push!(sSSE, sum(abs2, Y .- s.(aX * w)))
    push!(sENTROPY, sum(-Y .* log.(s.(aX * w)) .- (1 .- Y) .* log.(1 .- s.(aX * w))))
end
sum(abs2, Y .- s.(aX * w))
pX = s.(aX * w) .|> round
plot(X[:, 1], X[:, 2], group = pX, seriestype = :scatter, legend = false)
sum(abs2, Y .- pX)

plot(sSSE, label = "SSE")
plot!(sENTROPY, label = "Entropy")

sw = deepcopy(w)

# Gradient Descent

SSE = []
ENTROPY = []
w = zeros(3)
for _ in 1:100
    # cross entropy
    w = w - .05 * Zygote.gradient(w -> sum(-Y .* log.(s.(aX * w)) .- (1 .- Y) .* log.(1 .- s.(aX * w))), w)[1]
    push!(SSE, sum(abs2, Y .- s.(aX * w)))
    push!(ENTROPY, sum(-Y .* log.(s.(aX * w)) .- (1 .- Y) .* log.(1 .- s.(aX * w))))
end
sum(abs2, Y .- s.(aX * w))
pX = s.(aX * w) .|> round

plot(X[:, 1], X[:, 2], group = pX, seriestype = :scatter, legend = false)
sum(abs2, Y .- pX)

plot(SSE, label = "SSE")
plot!(ENTROPY, label = "Entropy")

w, sw

plot(SSE, label = "SSE")
plot!(sSSE, label = "sSSE")



# Stocastic surrogate gradient descent

w = zeros(3)
for _ in 1:300
    i = rand(1:size(X)[1])
    w .+= dw(w, aX[i, :], Y[i])
end
sum(abs2, Y .- s.(aX * w))
pX = s.(aX * w) .|> round
plot(X[:, 1], X[:, 2], group = pX, seriestype = :scatter, legend = false)
sum(abs2, Y .- pX)

sloss(y, yh) = (1 - 2*y)*((2*y - 1)*log(exp(yh) + 1) - y^2 * yh) + (2*y - 1)/(exp(yh) + 1)

sloss.(Y, aX * w)

sseloss(y, yh) = (y - yh)^2
closs(y, yh) = -y*log(s(yh)) - (1 - y)*log(1 - s(yh))

w


sw1l(w1) = sum(sloss.(Y, aX * [w1; -3.8; 5.756]))
sl = sw1l.(-10:.01:10)
cw1l(w1) = sum(closs.(Y, aX * [w1; -3.8; 5.756]))
cl = cw1l.(-10:.01:10)
ssew1l(w1) = sum(abs2, Y .- s.(aX * [w1; -3.8; 5.756]))
ssel = ssew1l.(-10:.01:10)

argmax(sl), argmin(cl), argmin(ssel)
maximum(sl), minimum(cl), minimum(ssel)



# plot(w1 -> sum(sloss.(Y, aX * [w1; -3.8; 5.756]))/sum(closs.(Y, aX * [w1; -3.8; 5.756])), xlims=(-100, 100))
