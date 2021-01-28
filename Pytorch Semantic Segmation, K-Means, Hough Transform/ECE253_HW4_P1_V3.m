%hough transform (HT) using the (rho, theta) parameterization
%use accumulator cells w/ res of 1 deg in theta and 1 pixel in rho
%% Part 1 and 2

thetaRes = 1;
rhoRes = 1;
dim = 11;
testIm = zeros(dim, dim);
xs = [1, 11, 6, 1, 11];
ys = [1, 1, 6, 11, 11];
testIM(1, 1) = 1;
testIM(1, 11) = 1;
testIM(6, 6) = 1;
testIM(11, 1) = 1;
testIM(11, 11) = 1;
image = testIM;
figure()
imshow(image)
hold on
title('Original Image')
hold off

[HAccum, rows, thetaRad] = HT(image, thetaRes, rhoRes);



threshold = 2;

Ploot(image, threshold, rows, thetaRad, HAccum);

%% Part 3

image = imread('HW4_2020/data/lane.png');
image = imresize(image, .1);
figure()
imshow(image)
title('Original Image')

BW = rgb2gray(image);
imageTemp = edge(BW,'sobel');
image = imageTemp;
figure()
imshow(image)
title('Binary Edge Image')

[HAccum, rows, thetaRad] = HT(image, thetaRes, rhoRes);
threshold = 0.75*max(HAccum(:));
Ploot(image, threshold, rows, thetaRad, HAccum);


%% Part 4
image = imread('HW4_2020/data/lane.png');
image = imresize(image, .1);


BW = rgb2gray(image);
imageTemp = edge(BW,'sobel');
image = imageTemp;


[HAccum, rows, thetaRad] = HT(image, thetaRes, rhoRes);

threshold = 0.75*max(HAccum(:));
Ploot(image, threshold, rows, thetaRad, HAccum);


function [HAccum, rows, thetaRad] = HT(image, thetaRes, rhoRes)
sz = size(image);
dist = floor(sqrt( sz(1)*sz(1) + sz(2)*sz(2) ));
theta = (-90:thetaRes:90);
thetaRad = theta.*(pi/180);
rows = (-dist:rhoRes:dist);
HAccum = zeros(length(rows), length(thetaRad));
cosTheta = cos(thetaRad);
sinTheta = sin(thetaRad);
[yInd, xInd] = find(image~=0); %y is pts(1), x pts(2)
numPts = length(xInd);
figure()
for i = 1:numPts
 i;
 %yInd = pts(1);
 %xInd = pts(2);
 add = dist+1;
 for j = 1:length(thetaRad)
    y = yInd(i);
    x = xInd(i);
    r = x*cosTheta(j) + y*sinTheta(j);
    rowA = floor(r) + add;
    rTemp(j) = r;
    HAccum(rowA,j) = 1+ HAccum(rowA,j);
 end
 plot(theta, -rTemp)
 hold on
 colorbar
 xlabel('theta')
 ylabel('rho')
 grid on
 nothing = 0;
end
rows = (-dist:rhoRes:dist);
hold off
end







function Ploot(image, threshold, rows, thetaRad, HAccum)
numPts = length(rows);    
sz = size(HAccum);
    
linePts = (1:numPts);
lineAxs=zeros(size(linePts));
countLines =0;
for i=1:sz(1)
 for j=1:sz(2)
  if HAccum(i,j)>=threshold+1
    countLines =countLines+1;
    lineAxs(countLines,:)=(rows(i)-linePts*cos(thetaRad(j)))/sin(thetaRad(j));
    end
 end
end

figure()
imshow(image)
hold on
for l = 1:countLines
   plot(linePts, lineAxs(l,:), 'b')
   hold on
end
title('Image with Hough Lines')
hold off
end




















