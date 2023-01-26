using Zygote, MLJ, Plots, Images
gr()
X, y = make_moons(1000, noise=0.1, as_table = false)

Y = 2 * y .- 1


clf(w, b, x) = begin
    ((x .- w) .^ 2)'*(b .^ -2) - 100
end

clfdw(w, b, x) = -2 * (x - w) .* (b .^ -2)
clfdb(w, b, x) = -2 * ((x - w) .^ 2) .* (b .^ -3)

lossdw(w, b, x, y) = (y*clf(w, b, x) < 1) * clfdw(w, b, x)
lossdb(w, b, x, y) = (y*clf(w, b, x) < 1) * clfdb(w, b, x)


function loss(w, b, X, y)
    sum(max(0, 1 - clf(w, b, X[i, :]) * y[i]) for i in 1:size(y, 1))
end

function sgdloss(w, b, x, y)
    max(0, 1 - clf(w, b, x) * y)
end

function grad(w, b, X, y)
    Zygote.gradient((w, b) -> loss(w, b, X, y), w, b)
end

function sgdgrad(w, b, x, y)
    Zygote.gradient((w, b) -> sgdloss(w, b, x, y), w, b) 
end


sink(x, y, s) = sin(-1 * norm(x .- y) .^ 2 / s)
cas(x) = sin(x) + cos(x)
kern(x, y) = exp(sin(norm(x .- y) * 5))


inds = rand(1:1000, 5)

K = [kern(X[i, :], X[j, :]) for i in eachindex(Y), j in inds]

X
K
begin
    losses = []
    w = ones(size(K, 2))
    b = ones(size(K, 2))
    # R = rand(1)
    
    for _ in 1:100
        push!(losses, loss(w, b, K, Y))
        i = rand(1:size(K, 1))
        x = K[i, :]
        y = Y[i]
        g = sgdgrad(w, b, x, y)
        w -= 0.005 * g[1]
        b -= 0.005 * g[2]
    end
end


w
b

plot(losses[1:end])

scatter(X[:, 1], X[:, 2], color = Y)

scatter(X[:, 1], X[:, 2], zcolor =[clf(w, b, K[i, :]) for i in eachindex(Y)])
# scatter(X[:, 1], X[:, 2], group =[clf(w, b, K[i, :]) for i in eachindex(Y)] .> 100)
scatter(w, b, legend=false)
begin
    img = [clf(w, b, [kern([i; j], X[k, :]) for k in inds] ) for i in -2:.05:2, j in -2:.05:2]
    (img .- minimum(img)) ./ (maximum(img)  - minimum(img)) .|> Gray
end



# scatter([clf(w, b, K[i, :]) for i in eachindex(Y)], Y, color = Y)
