module fic

using FileIO, Images, Interpolations

# Convert RGB image to grayscale
function grayscale(image::Array{RGB{Float64},2}) 
    img_planes = permutedims(channelview(image), (2,3,1));
    return Gray.(img_planes[:,:,1]*0.299 + img_planes[:,:,1]*0.587 + img_planes[:,:,1]*0.114);
end

# Block-based averaging to reduce size, this is slower than using imresize with a Linear interpolation scheme
function reduce(image::Array{Float64,2}, r::Int)
    out = zeros(div.(size(image),r));
    for i in axes(image,1), j in axes(image,2)
        out[i,j] = sum(image[r*(i-1)+1:r*i,r*(j-1)+1:r*j])/(r*r);
    end
    return out;
end

# Using imresize to reduce the size of blocks. Ensures mean stays the same before and after resizing.
# This ~4x faster than the block based averaging when using linear, and 4x slower when using cubic splines
function reduce(image::Array{Float64,2}, dims::Tuple{Int, Int}; method = Linear())
    out = imresize(image, dims, method = method);
    offset = sum(image)/prod(size(image)) - sum(out)/prod(size(out));
    clip!(out .+ offset)
    return out;
end

# Trims pixel values above 1 and below 0
function clip!(image::Array{Float64,2})
    for i in axes(image,1), j in axes(image,2)
        image[i,j] = min(max(image[i,j], 0), 1)
    end
end

# Finds the least squares solution to contrast*S + brightness = D
function find_contrast_and_brightness(D::Vector{Float64},S::Vector{Float64})::Tuple{Float64, Float64, Float64}
    A = hcat(ones(size(S)),S);
    x = (A'*A)\(A'*D);
    d::Float64 = 0;
    for i in eachindex(S)
        d += (S[i]*x[2] + x[1] - D[i])^2;
    end
    return d, x[1], x[2];
end

# Naive block based fractal compression on a greyscale image
function compress(image::Array{Gray{Float64},2}, s_size::Int64, d_size::Int64)
    image = channelview(image);
    n_s_blocks = size(image).÷s_size;
    n_d_blocks = size(image).÷d_size;
    reduced_blocks = Vector{Tuple{Int16,Int16,Bool,Int16,Vector{Float64}}}(undef, prod(n_s_blocks)*8);
    i = 1
    for k = 1:n_s_blocks[1], l = 1:n_s_blocks[2]
        S = fic.reduce(image[s_size*(k-1)+1:s_size*k,s_size*(l-1)+1:s_size*l], (d_size,d_size));
        for α in [0,90,180,270]
            T = rotr90(S, α÷90);
            reduced_blocks[i] = (k, l, false, α, vec(T));
            reduced_blocks[i+1] = (k, l, true, α, vec(T[end:-1:1,:]));
            i += 2;
        end
    end
    transformations = Matrix{Tuple{Int16,Int16,Bool,Int16,Float64,Float64}}(undef, n_d_blocks);
    for i in 1:n_d_blocks[1]
        for j in 1:n_d_blocks[2]
            min_d = Inf
            D = vec(image[d_size*(i-1)+1:d_size*i,d_size*(j-1)+1:d_size*j]);
            for (k, l, flip, α, S) in reduced_blocks
                d, brightness, contrast = find_contrast_and_brightness(D,vec(S[:]));
                if d < min_d
                    min_d = d;
                    transformations[i,j] = (k,l,flip,α,contrast,brightness);
                end
            end
        end
    end
    return transformations;
end

# Naive block based fractal decompression for greyscale image
# Returns a vector of images for each iteration of the transformation
function decompress(transformations::Matrix{Tuple{Int16,Int16,Bool,Int16,Float64,Float64}}, s_size::Int64, d_size::Int64, n_iter::Int64) 
    dims = size(transformations).*d_size;
    iterations = [rand(Gray{Float64},dims)];
    for it in 1:n_iter
        curr_iter = channelview(iterations[end]);
        next_iter = zeros(dims);
        for i in axes(transformations,1), j in axes(transformations,2)
            k,l,flip,α,contrast,brightness = transformations[i,j];
            S = rotr90(fic.reduce(curr_iter[s_size*(k-1)+1:s_size*k,s_size*(l-1)+1:s_size*l], (d_size,d_size)),α÷90);
            flip && (S = S[end:-1:1,:]);
            next_iter[d_size*(i-1)+1:d_size*i,d_size*(j-1)+1:d_size*j] = S*contrast .+ brightness;
        end
        push!(iterations, Gray.(next_iter));
    end
    return iterations
end

function compress_RGB(image::Array{RGB{Float64},2}, s_size::Int64, d_size::Int64)
    return compress.([Gray.(channelview(image)[i,:,:]) for i in 1:3], s_size, d_size);
end

function decompress_RGB(transformations::Vector{Matrix{Tuple{Int16,Int16,Bool,Int16,Float64,Float64}}}, s_size::Int64, d_size::Int64, n_iter::Int64)
    iters = decompress.(transformations, s_size, d_size, n_iter);
    return [colorview(RGB,iters[1][n], iters[2][n], iters[3][n]) for n in 1:n_iter];
end

end # module fic
