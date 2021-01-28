%%
%William Argus, A12802324

%%
% Part i
image = imread('circles_lines.jpg');
%image = image(:,:,1);
image=im2bw(image,50/255);
figure(1);
subplot(221);
imshow(image)
title('Original image');
axis on;

i=5;
circle = strel('disk',i);
postOpening = imopen(image,circle);

%figure(2);
axes();
subplot(222);
imshow(postOpening);
title('Image after opening');
axis on;

L = bwlabel(postOpening);
temp = max(max(L));

LAlt = L;
LAlt(LAlt == 0) = 31;
%figure(3);
subplot(223);
imshow(LAlt, colorcube);
h = colorbar;
set(h, 'ylim', [0 30])
title('Image after connected component labeling');
axis on;

LAlt = L;
LAlt(LAlt == 0) = 50;
%figure(4);
subplot(224);
imshow(LAlt, jet);
k = colorbar;
set(k, 'ylim', [0 30])
title({'Image after connected component labeling,', 'alternate colormap'});
axis on;

region = zeros(temp,1);
x_centroid = zeros(temp,1);
y_centroid = zeros(temp,1);
area = zeros(temp,1);
%region, x_centroid, y_centroid, area
sz=size(L);
sz=uint8(sz);
for i = 1:temp
    region(i,1)=i;
    listVals = find(L == i);
    area(i,1) = length(listVals);
    
    indics = zeros(length(listVals), 2); %x and y indices
    for j = 1:length(listVals) 
        x = idivide(listVals(j,1), sz(1));
        y = rem(listVals(j,1), sz(1));
        indics(j,1) = x;
        indics(j,2) = y; 
    end
    centroid = median(indics,1);
    x_centroid(i,1) = centroid(1);
    y_centroid(i,1) = centroid(2);
end

tabulate = table(region, x_centroid, y_centroid, area);
tabulate(:,:)

%%
%for i = 5:5
%    circle = strel('disk',i);
%    postOpening = imopen(image,circle);
%    figure(i);
%    imshow(postOpening);
%end

%Part ii
%close all;
%clear all;
image2 = imread('lines.jpg');
image2=im2bw(image2,50/255);
figure(5);
subplot(221);
imshow(image2)
title('Original image');
axis on;

line = strel('line', 9, 90);
postOpening2 = imopen(image2,line);

%figure(6);
axes();
subplot(222);
imshow(postOpening2);
title('Image after opening');
axis on;

L2 = bwlabel(postOpening2);
numLines = max(max(L2));

%figure(7);
%imshow(L2, lines);
%h = colorbar;
%set(h, 'ylim', [-1 6])
%title('Image after connected component labeling');
%axis on;

L2Alt = L2;
L2Alt(L2Alt == 0) = 7;
%figure(8);
subplot(223);
imshow(L2Alt, lines);
k = colorbar;
set(k, 'ylim', [1 7])
title('Image after connected component labeling');
axis on;


region2 = zeros(numLines,1);
x_centroid2 = zeros(numLines,1);
y_centroid2 = zeros(numLines,1);
area2 = zeros(numLines,1);
%region, x_centroid, y_centroid, area
sz2=size(L2);
sz2=uint8(sz2);
for i = 1:numLines
    region2(i,1)=i;
    listVals2 = find(L2 == i);
    area2(i,1) = length(listVals2);
    
    indics2 = zeros(length(listVals2), 2); %x and y indices
    for j = 1:length(listVals2) 
        x = idivide(listVals2(j,1), sz2(1));
        y = rem(listVals2(j,1), sz2(1));
        indics2(j,1) = x;
        indics2(j,2) = y; 
    end
    centroid2 = median(indics2,1);
    x_centroid2(i,1) = centroid2(1);
    y_centroid2(i,1) = centroid2(2);
end

tabulate2 = table(region2, x_centroid2, y_centroid2, area2);
tabulate2(:,:)