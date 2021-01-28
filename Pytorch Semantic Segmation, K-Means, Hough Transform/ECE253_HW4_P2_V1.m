% N is number of pixels in image
% M = 3 for RGB
% k is number of clusters


%setup first function
im = imread('HW4_2020/data/white-tower.png');
sz = size(im);
N = sz(1)*sz(2);
M = 3;

%call first function
features = createDataset(im);

%setup second function
nclusters = 7;
rng(5);
id = randi(size(features, 1), 1, nclusters);
centers = features(id, :);

%call second function 
[idx, centers] = kMeansClustering(features, centers);

%call third function
im_seg = mapValues(im, idx, centers);

%display image 
figure (1)
imshow(im)
title('image before segmentation', 'FontSize', 24);
%display image seg
figure (2)
imshow(im_seg)
title('image after segmentation', 'FontSize', 24);

centers

%features = reshape(im, [N,M]);
function features = createDataset(im)
    sz = size(im);
    N = sz(1)*sz(2);
    M = 3;
    features = reshape(im, [N,M]);
end

function [idx, centers] = kMeansClustering(features, centers)
    %idx = zeros(N,1);
    %comannds: pdist2, find, check new center: sum of features in
    %class/points in that class
    newCenters = cast(centers, 'double');
     sz = size(centers);
     nclusters = sz(1);
    % FUNCTION 2
    %centers and features already defined
    %idx = zeros(N,1);
    for i = 1:100
        dist = pdist2(features, centers);
        [minDist, minIndex] = min(dist, [], 2);

        for k = 1:nclusters
           allPts = features( minIndex== k, : );
           tempCenter = mean(allPts,1);
           newCenters(k,:) = tempCenter;

        end
        if isequal(centers, newCenters)
            break
        end
        centers = newCenters;
    end
    idx = minIndex;
    nothing  =0;
end

function im_seg = mapValues(im, idx, centers)
    features = createDataset(im);
    sz = size(features);
    for l = 1:sz(1)
        k = idx(l);
        features(l,:) = centers(k, :);
    end
    %features(idx == k, :) = centers(k, :);
    im_seg = reshape(features, size(im) );
    nothing =0;
end
