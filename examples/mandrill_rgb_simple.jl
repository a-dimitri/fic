using fic, FileIO, Images, TestImages, Plots, Plots.PlotMeasures

data = imresize(float64.(testimage("mandrill")), (256,256));

source_size = 8;
destination_size = 4;
transforms = fic.compress_RGB(data, source_size, destination_size);
n_iterations = 15;
iterations = fic.decompress_RGB(transforms, source_size, destination_size, n_iterations);

anim = @animate for im in iterations
    plot(im,axis=([],false),margins=0px,size=size(im));
end
    
gif(anim, "mandrill_rgb_fps4.gif", fps=4);

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

