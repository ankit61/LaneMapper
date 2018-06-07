% Evaluate.cpp makes its final results look like these: all pixels that are predicted to be road are colored red and all other pixels are black. 
% However, KITTI training set uses a different color scheme to show the data.  
% This scripts takes as input a directory which stores images labelled by KITTI and converts all those images to the same form that Evaluate.cpp would generate.
% This helps comparing corresponding images easier as both are in same format.

%change the strings in root and imwrite to customize for your own applications
clear all
root = 'C:\Users\Ankit\Documents\TAMU Succeed\Research\Autonomous Driving\data_road\training\gt_image_2\';
files = dir(root);
for i = 1:length(files)
    if(contains(files(i).name, 'road'))
        im = imread(char(join([root files(i).name], '')));
        im(:,:,1) = zeros(size(im, 1), size(im, 2));
        im(:,:,2) = zeros(size(im, 1), size(im, 2));
        im(:, :, 1) = im(:, :, 3);
        im(:,:,3) = zeros(size(im, 1), size(im, 2));
        imwrite(im, ['C:\Users\Ankit\Documents\TAMU Succeed\Research\Autonomous Driving\segmented\truth\segmented_' files(i).name]);
    end
end
