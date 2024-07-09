module fic

using FileIO, Images, Interpolations

# Read image into array of floats of size (H,W,P)
function imread(path::String) 
    return Float64.(permutedims(channelview(load(path)),(2,3,1)));
end

# Convert (H,W,3) RGB image into (H,W) greyscale image
function greyscale(image::Array{<:AbstractFloat,3}) 
    return image[1,:,:]*0.299 + image[2,:,:]*0.587 + image[3,:,:]*0.114;
end

# Block-based averaging to reduce size, this is slower than using imresize with a Linear interpolation scheme
function reduce(image::Array{<:AbstractFloat,2}, r::Int)
    out = zeros(div.(size(image),r));
    for i in axes(out,1)
        for j in axes(out,2)
            out[i,j] = sum(image[r*(i-1)+1:r*i,r*(j-1)+1:r*j])/(r*r);
        end
    end
    return out;
end

# Using imresize to reduce the size of blocks. Ensures mean stays the same before and after resizing.
# This ~4x faster than the block based averaging when using linear, and 4x slower when using cubic splines
function reduce(image::Array{<:AbstractFloat,2}, dims::Tuple{Int, Int}; method = Linear())
    out = imresize(image, dims, method = method);
    offset = sum(image)/prod(size(image)) - sum(out)/prod(size(out));
    clip!(out .+ offset)
    return out;
end

# Trims pixel values above 1 and below 0
function clip!(image::Array{<:AbstractFloat,2})
    for i in axes(image,1)
        for j in axes(image,2)
            image[i,j] = min(max(image[i,j], 0), 1)
        end
    end
end

# Naive block based fractal compression on a greyscale image
function compress(image::Array{<:AbstractFloat,2}, D_size, R_size)
    transformations = []
    
    return transformations
end

end # module fic
