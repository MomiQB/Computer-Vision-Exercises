%% read images
elephantImage = imread("./immaginiObjectDetection/elephant.jpg");
sceneImage = imread("./immaginiObjectDetection/clutteredDesk.jpg");

%figure(1), clf, imshow(elephantImage)
%figure(2), clf, imshow(sceneImage)

%figure(1), clf, imagesc(elephantImage)
%figure(2), clf, imagesc(sceneImage)

% compute the scale factor so that we can perform sliding window woth a
% fixed scale. we compute it (manually) as the ratio of thr same box
% dimension in the two image 

% option 1: no scale factor
fs = 1; 
% option 1: slightly increase size 
fs = 0.9; 

elephantImage = imresize(elephantImage, 1/fs);
%figure(3), clf, imshow(elephantImage)

%% sliding window
% elephantImage = im2double(elephantImage);  %converts in 0-1
% sceneImage = im2double(sceneImage);
% Sb=size(elephantImage);
% Ss=size(sceneImage);
% step=5;                                   % play with parameter
% Map=[];
% 
% for rr=1:step:Ss(1)-Sb(1)
%    tmp=[];
%    for cc=1:step:Ss(2)-Sb(2)
%        D=sceneImage(rr:rr+Sb(1)-1,cc:cc+Sb(2)-1)-elephantImage;
%        D=D.^2;
%        D=sum(D,"all");     % <----- SSD
%       tmp=[tmp D];
%    end
%    Map=[Map; tmp];
%    figure(3), clf, imagesc(Map), colorbar, drawnow    %blue where low square difference, yellow where high
% end                                                    %no clear spot of where the image is
% toc

clear	         % clear workspace	
close all        % clear figures
clc              % clear command window

elephantImage = imread("./immaginiObjectDetection/elephant.jpg");
sceneImage = imread("./immaginiObjectDetection/clutteredDesk.jpg");

%% key point detection
tic
elephantPoints = detectSURFFeatures(elephantImage, "MetricThreshold", 800);         % alternative to SIFT saw in class
scenePoints = detectSURFFeatures(sceneImage, "MetricThreshold",800);

% plot the 100 strongest keypoints for template image
figure(1), clf
imshow(elephantImage), hold on
plot(selectStrongest(elephantPoints, 100)), hold on   

% plot the 100 strongest keypoints for scene image
figure(2), clf
imshow(sceneImage), hold on
plot(selectStrongest(scenePoints, 100)), hold off

%% key point description
[elephantFeatures, elephantPoints] = extractFeatures(elephantImage, elephantPoints);
[sceneFeatures, scenePoints]=extractFeatures(sceneImage, scenePoints);

%% feature matching
boxPairs = matchFeatures(elephantFeatures, sceneFeatures,...
    "MatchThreshold", 40, "MaxRatio",0.8);
matchedelephantPoints = elephantPoints(boxPairs(:,1),:);
matchedScenePoints = scenePoints(boxPairs(:,2),:);

% plot the matched features between the 2 images
figure(3),clf
showMatchedFeatures(elephantImage, sceneImage, matchedelephantPoints,...
	matchedScenePoints, 'montage');


%% geometric consistency check
[tform, inlierelephantPoints, inlierScenePoints]=...
	estimateGeometricTransform(matchedelephantPoints,...
	matchedScenePoints, 'affine', Confidence = 80);

% plot the matched features between the 2 images after geometric
% consistency check
figure(4),clf
showMatchedFeatures(elephantImage, sceneImage, inlierelephantPoints,...
	inlierScenePoints, 'montage');



%% bounding box drawing
% first option

boxPoly = [1 1;
    size(elephantImage,2) 1;
    size(elephantImage,2) size(elephantImage,1);
    1 size(elephantImage,1);
    1 1];

newBoxPoly = transformPointsForward(tform, boxPoly);

% plot the bounding box on the scene image
figure(5), clf
imshow(sceneImage), hold on
line(newBoxPoly(:,1), newBoxPoly(:,2),'Color','y')
hold off
toc
%% bounding polygon drawing (more specific)

boxPoly = [295.6885   90.5387;
  289.2341   29.8675;
  321.5060   15.6678;
  342.1601   44.0671;
  324.0878   62.1394;
  336.9966  148.6282;
  315.0516  186.0637;
  320.2151  204.1360;
  284.0706  218.3356;
  322.7969  342.2599;
  109.8021  343.5508;
   50.4217  312.5697;
   46.5491  273.8434;
   50.4217  171.8640;
   72.3666  120.2289;
  114.9656   95.7022;
  184.6730   89.2478;
  224.6902   78.9208;
  251.7986   60.8485;
  272.4527   68.5938;
  295.6885   90.5387
];

newBoxPoly = transformPointsForward(tform, boxPoly);

% plot the bounding polygon on the scene image
figure(6), clf
imshow(sceneImage), hold on
line(newBoxPoly(:,1), newBoxPoly(:,2),'Color','y')
hold off

%% more precise bounding box (manual)
% second option

%figure('WindowStyle', 'docked'), clf
%imshow(elephantImage) 
%[x,y]=ginput(20);


%x = [x; x(1)];
%y = [y; y(1)];
%newBoxPoly = transformPointsForward(tform, [x,y]);
%figure, clf
%imshow(sceneImage), hold on
%line(newBoxPoly(:,1), newBoxPoly(:,2),'Color','y')
%hold off

%figure('WindowStyle', 'docked'), clf, imshow(elephantImage)



