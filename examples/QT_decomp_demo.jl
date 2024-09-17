using fic, FileIO, Images, TestImages, Plots, Plots.PlotMeasures

data = float64.(testimage("cameraman"));
image = copy(channelview(data));

function quad_tree_image(decomp::Vector{Tuple{Int16,Int16,Int16,Int16,Float64}},
                        image_size::Tuple{Int64,Int64})
    qt_im = zeros(image_size);
    for (x, y, h, w, u) in decomp
        for i in x:x+h-1
            for j in y:y+w-1
                qt_im[i,j] = u
            end
        end
    end
    return Gray.(qt_im);
end

data

decomp = fic.quad_tree_decomp(image, (2,2), 0.002);
size(decomp)
QT_image = quad_tree_image(decomp, size(image))

decomp = fic.quad_tree_decomp(image, (2,2), 0.01);
size(decomp)
QT_image = quad_tree_image(decomp, size(image))

decomp = fic.quad_tree_decomp(image,(2,2), 0.03);
size(decomp)
QT_image = quad_tree_image(decomp, size(image))

decomp = fic.quad_tree_decomp(image, (2,2), 0.05);
size(decomp)
QT_image = quad_tree_image(decomp, size(image))

decomp = fic.quad_tree_decomp(image, (2,2), 0.10);
size(decomp)
QT_image = quad_tree_image(decomp, size(image))