clear
clc

videoID = 1449;
r = 425;
c = 560;

csp = rand(10000,3)*255;  % color panel
csp(1,:) = 0;

files = dir(sprintf('./correspondences/%04d',videoID));

for i = 3:length(files)
    name = files(i).name;
    corr = load(sprintf('./correspondences/%04d/%s',videoID,name));
    corr = reshape(corr,[r,c]);
    imshow(ind2rgb(corr, im2double(uint8(csp))));
    drawnow;
end