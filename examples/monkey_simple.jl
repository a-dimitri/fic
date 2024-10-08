using fic, FileIO, Images, Plots, Plots.PlotMeasures

data = float64.(load("sample_images/monkey.gif"));
data = fic.grayscale(data);

source_size = 8;
destination_size = 4;
transforms = fic.compress(data, source_size, destination_size);
n_iterations = 15;
iterations = fic.decompress(transforms, source_size, destination_size, n_iterations);

anim = @animate for im in iterations
    plot(im,axis=([],false),margins=0px,size=size(im));
end
    
gif(anim, "monkey_fps4.gif", fps=4);

plot(assess_ssim.(iterations, [data]), 
        linewidth = 2, 
        xlims=(1,16), 
        xlabel = "Iteration Number",
        ylims=(0,1), 
        ylabel = "SSIM",
        legend=false, 
        markershape=:x
)

plot(assess_psnr.(iterations, [data]), 
        linewidth = 2, 
        xlims=(1,16), 
        xlabel = "Iteration Number",
        ylabel = "PSNR (dB)",
        legend=false, 
        markershape=:x
)

