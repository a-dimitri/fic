module fic

using FileIO, Images, Interpolations, OffsetArrays

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

# Gives a vector of top left co-ordinates, width and height of each source block
function get_source_blocks(image::Array{Float64,2}, s_size::Int64)
    n_s_blocks = size(image).÷s_size;
    source_blocks = Vector{Tuple{Int16,Int16,Int16,Int16}}(undef, prod(n_s_blocks));
    for k = 1:n_s_blocks[1], l = 1:n_s_blocks[2]
        source_blocks[(k-1)*n_s_blocks[1] + l] = (k, l, s_size, s_size);
    end
    return source_blocks;
end

# Redues all the source blocks to the size of the destination blocks
function get_reduced_blocks(image::Array{Float64,2}, source_blocks::Vector{Tuple{Int16,Int16, Int16,Int16}}, d_size::Int64)
    reduced_blocks = Vector{Tuple{Int16,Int16,Int16,Int16,Bool,Int16,Vector{Float64}}}(undef, length(source_blocks)*8);
    i = 1
    for (k, l, h, w) in source_blocks
        S = fic.reduce(image[h*(k-1)+1:h*k,w*(l-1)+1:w*l], (d_size,d_size));
        for α in [0,90,180,270]
            T = rotr90(S, α÷90);
            reduced_blocks[i] = (k, l, h, w, false, α, vec(T));
            reduced_blocks[i+1] = (k, l, h, w, true, α, vec(T[end:-1:1,:]));
            i += 2;
        end
    end
    return reduced_blocks;
end

# Prefix map-sum an image with a map supplied in func
function prefix_sum_image(image, func=x->x) 
    pfx_sum = OffsetArray(zeros(size(image) .+ (1,1)), -1, -1);
    for i in axes(image,1)
        curr_sum = 0.0
        for j in axes(image,2)
            curr_sum += func(image[i,j]);
            pfx_sum[i,j] = pfx_sum[i-1,j] + curr_sum;
        end
    end
    return pfx_sum;
end

# Prefix compute the map-sum of a region of an image using a precomputed prefix sum
function region_sum(pfx_sum, x2, y2, x1, y1)
    return pfx_sum[x1,y1] - pfx_sum[x1,y2-1] - pfx_sum[x2-1,y1] + pfx_sum[x2-1,y2-1];
end

# Recursive helper used by quad_tree_decomp
function _quad_tree_decomp(image::Array{Float64,2}, 
                        pfx_sum::OffsetMatrix{Float64, Matrix{Float64}}, 
                        pfx_sum2::OffsetMatrix{Float64, Matrix{Float64}}, 
                        top_left::Tuple{Int64,Int64}, 
                        curr_size::Tuple{Int64,Int64}, 
                        size_limit::Tuple{Int64,Int64},
                        std_threshold::Float64,
                        decomp::Vector{Tuple{Int16, Int16, Int16, Int16, Float64}})
    bottom_right = top_left .+ curr_size .- (1,1);
    mean_x = region_sum(pfx_sum, top_left[1], top_left[2], bottom_right[1], bottom_right[2])/prod(curr_size);
    mean_x2 = region_sum(pfx_sum2, top_left[1], top_left[2], bottom_right[1], bottom_right[2])/prod(curr_size);
    var = mean_x2 - (mean_x)^2;
    if var < std_threshold^2 || curr_size == size_limit || curr_size.%2 != (0,0)
        push!(decomp, (Int16.(top_left)..., Int16.(curr_size)..., mean_x));
    else
        new_size = curr_size.÷2;
        new_top_left = top_left .+ new_size;
        _quad_tree_decomp(image, pfx_sum, pfx_sum2, top_left, new_size, size_limit, std_threshold, decomp);
        _quad_tree_decomp(image, pfx_sum, pfx_sum2, (top_left[1], new_top_left[2]), new_size, size_limit, std_threshold, decomp);
        _quad_tree_decomp(image, pfx_sum, pfx_sum2, (new_top_left[1], top_left[2]), new_size, size_limit, std_threshold, decomp);
        _quad_tree_decomp(image, pfx_sum, pfx_sum2, new_top_left, new_size, size_limit, std_threshold, decomp);
    end
end

# Perform an efficient quad tree decomposition, using a prefix sum
# Runs in O(hw) time on an image with height h and width w
function quad_tree_decomp(image::Array{Float64,2},
                        size_limit::Tuple{Int64,Int64},
                        std_threshold::Float64)
    pfx_sum = prefix_sum_image(image);
    pfx_sum2 = prefix_sum_image(image, x->x^2);
    decomp = Vector{Tuple{Int16,Int16,Int16,Int16,Float64}}();
    _quad_tree_decomp(image, pfx_sum, pfx_sum2, (1,1), size(image), size_limit, std_threshold, decomp);
    return decomp;
end

# Block based fractal compression on a greyscale image
# Destination blocks are selected naively, while source blocks can be selected using different methods
function compress(image::Array{Gray{Float64},2}, s_size::Int64, d_size::Int64)
    image = copy(channelview(image));
    source_blocks = fic.get_source_blocks(image, s_size);
    reduced_blocks = fic.get_reduced_blocks(image, source_blocks, d_size);
    n_d_blocks = size(image).÷d_size;
    transformations = Matrix{Tuple{Int16,Int16,Int16,Int16,Bool,Int16,Float64,Float64}}(undef, n_d_blocks);
    for i in 1:n_d_blocks[1]
        for j in 1:n_d_blocks[2]
            min_d = Inf
            D = vec(image[d_size*(i-1)+1:d_size*i,d_size*(j-1)+1:d_size*j]);
            for (k, l, h, w, flip, α, S) in reduced_blocks
                d, brightness, contrast = find_contrast_and_brightness(D,vec(S[:]));
                if d < min_d
                    min_d = d;
                    transformations[i,j] = (k,l,h,w,flip,α,contrast,brightness);
                end
            end
        end
    end
    return transformations;
end

# Block based fractal decompression for greyscale image
# Returns a vector of images for each iteration of the transformation
function decompress(transformations::Matrix{Tuple{Int16,Int16,Int16,Int16,Bool,Int16,Float64,Float64}}, s_size::Int64, d_size::Int64, n_iter::Int64) 
    dims = size(transformations).*d_size;
    iterations = [rand(Gray{Float64},dims)];
    for it in 1:n_iter
        curr_iter = channelview(iterations[end]);
        next_iter = zeros(dims);
        for i in axes(transformations,1), j in axes(transformations,2)
            k,l,h,w,flip,α,contrast,brightness = transformations[i,j];
            S = rotr90(fic.reduce(curr_iter[h*(k-1)+1:h*k,w*(l-1)+1:w*l], (d_size,d_size)),α÷90);
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

function decompress_RGB(transformations::Vector{Matrix{Tuple{Int16,Int16,Int16,Int16,Bool,Int16,Float64,Float64}}}, s_size::Int64, d_size::Int64, n_iter::Int64)
    iters = decompress.(transformations, s_size, d_size, n_iter);
    return [colorview(RGB,iters[1][n], iters[2][n], iters[3][n]) for n in 1:n_iter];
end

end # module fic
