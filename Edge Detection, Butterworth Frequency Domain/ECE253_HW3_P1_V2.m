%% ECE253, HW3, Problem 1, Version 2
% William Argus A12802324
%% setup
clc;
clear all;
close all;

image = imread('geisel.jpg');
imageGray = rgb2gray(image);
te = 130
edges = cannyEdgeDetection(imageGray, te);
figure(3);
imshow(im2uint8(edges/255)); 
title('Final edge image after thresholding', 'FontSize', 16);
%% function
function imageEdges = cannyEdgeDetection(imageGray, te)
    %% Part 1
    k = (1/159)*[2 4 5 4 2; 4 9 12 9 4; 5 12 15 12 5; 4 9 12 9 4; 2 4 5 4 2];
    imageSmooth = conv2(imageGray, k);
    imageSmooth = im2uint8(imageSmooth/255);

    %% Part 2

    kx = [-1 0 1; -2 0 2; -1 0 1];
    ky = [-1 -2 -1; 0 0 0; 1 2 1];

    gX = conv2(imageSmooth, kx);
    gY = conv2(imageSmooth, ky);
    absGradient = sqrt(gX.^2 + gY.^2);
    gX(gX == 0) = 0.0001;
    angleGradient = atand(gY./gX);
    figure(1);
    imshow(im2uint8(absGradient/255));
    title('Original gradient magnitude image', 'FontSize', 16);

    %% Part 3
    angleGradient(-22.5 <= angleGradient & 22.5 > angleGradient) = 0;
    angleGradient(-67.5 <= angleGradient & -22.5 > angleGradient) = -45;
    angleGradient(67.5 > angleGradient & 22.5 <= angleGradient) = 45;
    angleGradient(-67.5 > angleGradient) = 90;
    angleGradient(67.5 <= angleGradient) = 90;

    %for each pixel
    sz = size(absGradient);
    pixel1=0;
    pixel2=0;
    for i = 2:sz(1)-1
       for j = 2:sz(2)-1
           %get gradient direction to pick comparison pixels
           if angleGradient(i,j) == 90
               pixel1 = absGradient(i-1,j);
               pixel2 = absGradient(i+1,j);
           elseif angleGradient(i,j) == -45
               pixel1 = absGradient(i-1,j-1);
               pixel2 = absGradient(i+1,j+1);
           elseif angleGradient(i,j) == 0
               pixel1 = absGradient(i,j+1);
               pixel2 = absGradient(i,j-1);
           elseif angleGradient(i,j) == 45
               pixel1 = absGradient(i-1,j+1);
               pixel2 = absGradient(i+1,j-1);        
           else
               print("ERROR!!");
           end
           if absGradient(i,j) > pixel1 && absGradient(i,j) > pixel2
               nothing=0;
           else
               absGradient(i,j)=0;
           end
        end
    end
    
    absGradientNormed = normalize(absGradient);
    figure(2);
    imshow(im2uint8(absGradient/255));
    title('Image after NMS', 'FontSize', 16);
   
    %% Part 4

    absGradient(absGradient < te) =0;
    imageEdges = absGradient;
end
