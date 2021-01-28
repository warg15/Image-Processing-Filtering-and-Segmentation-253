%% ECE253, HW3, Problem 2ii, Version 1
% William Argus A12802324
%% setup

image = imread('Street.png');
figure(1);
imshow(image);
title('Unpadded original image')
colorbar; 

sz = size(image);
padsize1 = (512-sz(1))/2 +1;
padsize2 = (512-sz(2))/2 +1;
paddedImage = im2uint8(zeros(512,512));

paddedImage((padsize1+1):(padsize1+sz(1)), (padsize2+1):(padsize2+sz(2))) = image;
%figure(2);
%imshow(paddedImage);

imFFT = fft2(paddedImage);
%test = ifft2(imFFT);
imFFT = fftshift(imFFT);
%figure(4);
%imshow(im2uint8(test/255));
figure(2)
imagesc(-256:255,-256:255,log(abs(imFFT))); 
colorbar; 
title('2D DFT log-magnitude of original image');
xlabel('u'); 
ylabel('v');

[u,v] = meshgrid(-256:255);
%%
%iterate through all possible u values, then through all possible v values,
%then for each (u,v), calcuate Dk
%then Hnr for each (u,v)?
%since there is 4 bursts, need 4 pairs of coordinates for the bursts

%calculate Hnr
Hnr = ones(512,512);
Do = 50
n = 2
uv = [0, -165; 165, 0;]
for k = 1:2
    uk = uv(k,1);
    vk = uv(k,2);
    Dpk = ((u - uk).^2 + (v - vk).^2).^.5;
    Dmk = ((u + uk).^2 + (v + vk).^2).^.5;
    
    term1 = ( 1+ (Do.*(Dpk.^-1)).^(2*n) ).^-1;
    term2 = ( 1+ (Do.*(Dmk.^-1)).^(2*n) ).^-1;
    temp = term1.*term2;
    Hnr = Hnr.*temp;
end

figure(3);
imshow(Hnr);
colorbar; 
title('The butterworth Notch Reject Filter in frequency domain Hnr(u,v)');
h = gca;
set(h, 'Visible', 'On')

freqImage = Hnr.*imFFT;
figure(4);
imagesc(-256:255,-256:255,log(abs(freqImage)));
colorbar; 
title('The frequency domain of the image after filtering');
xlabel('u'); 
ylabel('v');

unshift = ifftshift(freqImage);
resultPadded = ifft2(unshift);
%unpad image
result = resultPadded((padsize1+1):(padsize1+sz(1)), (padsize2+1):(padsize2+sz(2)));
figure(5);
imshow(im2uint8(result/255));
colorbar; 
title('The final filtered image');
