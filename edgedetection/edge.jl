using Images, ImageFiltering, TestImages

function scale(img::AbstractArray{T, 2}) where T <: Number
    minval = minimum(img)
    maxval = maximum(img)
    return (img .- minval) ./ (maxval - minval)
end

img = testimage("cameraman")

imfilter(img, Kernel.sobel())

k1, k2 = Kernel.sobel()
k1img = imfilter(img, k1)
k2img = imfilter(img, k2)

k1img .|> Float64 |> scale .|> Gray