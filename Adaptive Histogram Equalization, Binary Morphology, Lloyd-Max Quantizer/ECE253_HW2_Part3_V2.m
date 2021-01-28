%%
% William Argus, A12802324

%%
% Part i

%image = imread('diver.tif');
image = imread('lena512.tif');
%image = histeq(image,256);
figure(10);
imshow(image);

s = [1,2,3,4,5,6,7];

quantMSE = zeros(length(s),1);

for i = 1:7
    startPoint = 256/(2^s(i));
    numBins = 2^s(i);
    numThresh = numBins-1;
    %create thresholds
    thresh = [0];
    for j = 1:numThresh+1
        thresh = [thresh (startPoint*j -1)];
    end

    %create bin values
    firstBin = thresh(2)/2;
    increment = thresh(2) +1;
    binVals = [firstBin];
    for k = 2:numBins
       binVals = [binVals (binVals(k-1)+increment) ];
    end

    imageQuant = double(image);
    sz=size(imageQuant);
    for x = 1:sz(1)
       for y = 1:sz(2)
           for l = 1:numBins
               %if l == 1
               %    if imageQuant(x,y) <= thresh(l+1)
               %        imageQuant(x,y) = binVals(l);
               %    end
               if imageQuant(x,y) > thresh(l)
                   if imageQuant(x,y) <= thresh(l+1)
                       imageQuant(x,y) = binVals(l);
                   end
               end
           end
       end
    end
    figure(i);
    plotImageQuant = uint8(imageQuant);
    imshow(plotImageQuant);

    imageError = double(image);
    imageError = mean(mean((double(image) - imageQuant).^2));
    quantMSE(i) = imageError;
end

%% lloyd portion

imageDouble = double(image);
[M,N] = size(imageDouble);
training_set = reshape(imageDouble,N*M,1);

lloydMSE = zeros(length(s),1);

for i = 1:7
    len = 2^i;
    [PARTITION, CODEBOOK, DISTORTION] = lloyds(training_set, len);
    lloydMSE(i) = DISTORTION;
end
   


%% plot portion

figure(11);
hold on;
a(1) = plot(s,lloydMSE, 'DisplayName','Lloyd-Max MSE','LineWidth',2.0);
a(2) = plot(s,quantMSE, 'DisplayName','Simple Quant MSE','LineWidth',2.0);
%title('MSE of Lloyd-Max and Simple Quantizer, diver','FontSize',12);
title('MSE of Lloyd-Max and Simple Quantizer, Lena','FontSize',12);
%title('MSE of Lloyd-Max and Simple Quantizer, diver, after global histogram equalization','FontSize',12);
%title('MSE of Lloyd-Max and Simple Quantizer, Lena, after global histogram equalization','FontSize',12);
xlabel('Bit Rate','FontSize',14) 
ylabel('MSE','FontSize',14)
grid on;
legend({}, 'Location','northeast', 'FontSize',16)
hold off;


 