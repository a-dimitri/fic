module fic

using FileIO, Images, Interpolations

# Read image into array of floats of size (H,W,P)
function imread(path::String) 
    return Float64.(permutedims(channelview(load(path)),(2,3,1)));
end

# Convert (H,W,3) RGB image into (H,W) greyscale image
function greyscale(image::Array{<:AbstractFloat,3}) 
    return image[:,:,1]*0.299 + image[:,:,1]*0.587 + image[:,:,1]*0.114;
end

# Block-based averaging to reduce size, this is slower than using imresize with a Linear interpolation scheme
function reduce(image::Array{<:AbstractFloat,2}, r::Int)
    out = zeros(div.(size(image),r));
    for i in axes(image,1), j in axes(image,2)
        out[i,j] = sum(image[r*(i-1)+1:r*i,r*(j-1)+1:r*j])/(r*r);
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
    for i in axes(image,1), j in axes(image,2)
        image[i,j] = min(max(image[i,j], 0), 1)
    end
end

# Finds the least squares solution to contrast*S + brightness = D
function find_contrast_and_brightness(D,S)
    A = hcat(ones(size(S)),S);
    x = (transpose(A)*A)\(transpose(A)*D);
    d = sum( (S*x[2] .+ x[1] - D).^2 );
    return d, x...;
end

# Naive block based fractal compression on a greyscale image
function compress(image::Array{<:AbstractFloat,2}, s_size, d_size)
    n_s_blocks = size(image).÷s_size;
    n_d_blocks = size(image).÷d_size;
    reduced_blocks = [fic.reduce(image[s_size*(i-1)+1:s_size*i,s_size*(j-1)+1:s_size*j], (d_size,d_size)) for i = 1:n_s_blocks[1], j = 1:n_s_blocks[2]];
    transformations = Matrix{Tuple{Int16,Int16,Bool,Int16,Float64,Float64}}(undef, n_d_blocks);
    for i in 1:n_d_blocks[1], j in 1:n_d_blocks[2]
        println("$(i)/$(n_d_blocks[1]) ; $(j)/$(n_d_blocks[2])");
        min_d = Inf
        D = image[d_size*(i-1)+1:d_size*i,d_size*(j-1)+1:d_size*j];
        for k in 1:n_s_blocks[1], l in 1:n_s_blocks[2], flip in [false, true], α in [0,90,180,270]
            S = rotr90(reduced_blocks[k,l],α÷90);
            
            flip && (S = S[end:-1:1,:]);
            d, brightness, contrast = find_contrast_and_brightness(D[:],S[:]);
            if d < min_d
                min_d = d;
                transformations[i,j] = (k,l,flip,α,contrast,brightness);
            end
        end
    end
    return transformations;
end

# Naive block based fractal decompression for greyscale image
# Returns a vector of images for each iteration of the transformation
function decompress(transformations::Matrix{Tuple{Int16,Int16,Bool,Int16,Float64,Float64}}, s_size, d_size, n_iter) 
    dims = size(transformations).*d_size;
    iterations = [rand(Float64,dims)];
    for it in 1:n_iter
        println("$(it)/$(n_iter)");
        next_iter = zeros(dims);
        for i in axes(transformations,1), j in axes(transformations,2)
            k,l,flip,α,contrast,brightness = transformations[i,j];
            S = rotr90(fic.reduce(iterations[end][s_size*(k-1)+1:s_size*k,s_size*(l-1)+1:s_size*l], (d_size,d_size)),α÷90);
            flip && (S = S[end:-1:1,:]);
            next_iter[d_size*(i-1)+1:d_size*i,d_size*(j-1)+1:d_size*j] = S*contrast .+ brightness;
        end
        push!(iterations, next_iter);
    end
    return iterations
end

end # module fic
