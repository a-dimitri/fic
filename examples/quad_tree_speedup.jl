using fic, FileIO, Images, TestImages, Plots, Plots.PlotMeasures, BenchmarkTools

data = imresize(float64.(testimage("cameraman")),(256,256));
source_size = 8;
destination_size = 4;
bs = map([0,0.01,0.03,0.05]) do t
    @btime fic.compress_quad_tree($data, $source_size, $t, $destination_size);
end;
    