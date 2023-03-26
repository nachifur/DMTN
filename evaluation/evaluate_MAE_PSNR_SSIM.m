% Note: 
% wang_cvpr2018, le_iccv2019: no imresize; 
% fu_cvpr2021: first imresize, then double; 
% zhu_aaai2022: first double, then imresize; 
% this code: MAE->fu_cvpr2021, psnr+ssim->zhu_aaai2022
%% compute MAE
clear;close all;clc

% mask path
maskdir = 'D:\ncfey\Desktop\istd_test\test_B\';
MD = dir([maskdir '/*.png']);

% output path
shadowdir = '\\172.16.31.6\program_results\diffusion\Ç°ÆÚÌ½Ë÷\res_denoising_diffusion_tempalte_input_condition_mask_noise1_timestep20\';  
SD = dir([shadowdir '/*.png']);

% gt path
freedir = 'D:\ncfey\Desktop\istd_test\test_C\'; 
FD = dir([freedir '/*.png']);


total_dist_all = 0;
total_pixel_all = 0;
total_dists = 0;
total_pixels = 0;
total_distn = 0;
total_pixeln = 0;
allmae=zeros(1,size(SD,1)); 
smae=zeros(1,size(SD,1)); 
nmae=zeros(1,size(SD,1)); 
ppsnr=zeros(1,size(SD,1));
ppsnrs=zeros(1,size(SD,1));
ppsnrn=zeros(1,size(SD,1));
sssim=zeros(1,size(SD,1));
sssims=zeros(1,size(SD,1));
sssimn=zeros(1,size(SD,1));
cform = makecform('srgb2lab');

for i=1:size(SD)
    %% Load
    sname = strcat(shadowdir,SD(i).name);
    fname = strcat(freedir,FD(i).name); 
    mname = strcat(maskdir,MD(i).name); 
    s0=imread(sname);
    f0=imread(fname);
    m0=imread(mname);
    
    %% PSNR+SSIM
    f = double(f0)/255;
    s = double(s0)/255;
    m = m0;
    s=imresize(s,[256 256]);
    f=imresize(f,[256 256]);
    m=imresize(m,[256 256]);
    
    nmask=~m;       % mask of non-shadow region
    smask=~nmask;   % mask of shadow regions
    
    ppsnr(i)=psnr(s,f);
    ppsnrs(i)=psnr(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
    ppsnrn(i)=psnr(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));
    sssim(i)=ssim(s,f);
    sssims(i)=ssim(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
    sssimn(i)=ssim(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));
    
    %% MAE
    s=imresize(s0,[256 256]);
    f=imresize(f0,[256 256]);
    m=imresize(m0,[256 256]);
    f = double(f)/255;
    s = double(s)/255;
    m = m;
    
    nmask=~m;       % mask of non-shadow region
    smask=~nmask;   % mask of shadow regions
    
    f = applycform(f,cform);    
    s = applycform(s,cform);

    % MAE, per image
    dist=abs((f - s));
    
    sdist=dist.*repmat(smask,[1 1 3]);
    sumsdist=sum(sdist(:));
    
    ndist=dist.*repmat(nmask,[1 1 3]);
    sumndist=sum(ndist(:));
    
    sum_all_dist = sum(dist(:));
    
    sumsmask=sum(smask(:));
    sumnmask=sum(nmask(:));
    
    sum_all_mask = size(f,1)*size(f,2);

    smae(i)=sumsdist/sumsmask;
    nmae(i)=sumndist/sumnmask;
    allmae(i)=sum(dist(:))/sum_all_mask;
    
    % MAE, per pixel
    total_dists = total_dists + sumsdist;
    total_pixels = total_pixels + sumsmask;
    
    total_distn = total_distn + sumndist;
    total_pixeln = total_pixeln + sumnmask;  
       
    total_dist_all = total_dist_all + sum_all_dist;
    total_pixel_all = total_pixel_all + sum_all_mask;
    
    disp(i);
end
% Note: mean(allmae)=total_dist_all/total_pixel_all
fprintf('PSNR(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(ppsnr),mean(ppsnrn),mean(ppsnrs));
fprintf('SSIM(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(sssim),mean(sssimn),mean(sssims));
fprintf('PI-Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(allmae),mean(nmae),mean(smae));
fprintf('PP-Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n\n',total_dist_all/total_pixel_all,total_distn/total_pixeln,total_dists/total_pixels);