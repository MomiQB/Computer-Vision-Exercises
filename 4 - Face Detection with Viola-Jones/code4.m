%% Negative class

neg = dir('./CaltechFaces/my_train_non_face_scenes/*.jpg');

%% Negative Class Augmentation


mkdir('./CaltechFaces/my2_train_non_face_scenes/')
outdir = './CaltechFaces/my2_train_non_face_scenes';

%{
for ii = 1:size(neg,1)
    % original images
    im = imread([neg(ii).folder filesep neg(ii,1).name]);
    imwrite(im,[outdir filesep neg(ii,1).name]); 
    
    % obtain file parts (path + name + extension)
    [pp,ff,ee] = fileparts(neg(ii).name); % Extract file parts

    % vertical mirroring
    im_flip = fliplr(im); 
    imwrite(im_flip,[outdir filesep ff '_flip' ee]); 
    
    % horizontal mirroring 
    im_flip = flipud(im); 
    imwrite(im_flip,[outdir filesep ff '_flipud' ee]); 

    % rotation
    for n = 1:10
        angle = 360*rand(1); % generate a random angle between 0 and 360 degrees
        imr = imrotate(im, angle, 'crop');  % rotated image
        imwrite(imr, [outdir filesep ff '_r' num2str(n) ee]);
    end
    
    % brightness adjustment 
    for n = 1:6
        randomOffset = randi([-50, 50]); % random brightness offset between -50 and 50
        im_bright = im + randomOffset; % image with brightness adjustment
        imwrite(im_bright, [outdir filesep ff '_b' num2str(n) ee]); 
    end
    
    % Salt & Pepper noise addition
    for n = 1:6
        noiseDensity = randi([5, 20])/100; % random noise level between 0.05 and 0.20
        im_noisy = imnoise(im, 'salt & pepper', noiseDensity); % noisy image
        imwrite(im_noisy, [outdir filesep ff '_s&p' num2str(n) ee]); 
    end

    % camera motion addition
    for n = 1:6
        linMotion = randi([5, 20]); % random linear motion of camera between 5 and 20
        angleMotion = randi([0, 30]);  % random angle of camera motion between 0 and 30
        h = fspecial('motion', linMotion, angleMotion);  % create the filter
        im_motion = imfilter(im, h); % filtered image
        imwrite(im_motion, [outdir filesep ff '_mot' num2str(n) ee]); % save filtered image
    end
end

%}

%%

negativeFolder = './CaltechFaces/my2_train_non_face_scenes';
negativeImages = imageDatastore(negativeFolder);

%% positive class
faces = dir('./CaltechFaces/my_train_faces/*.jpg');
sz = [size(faces,1) 2];
varTypes = {'cell','cell'};
varNames = {'imageFilename','face'};
facesIMDB = table('Size',sz,'VariableTypes',varTypes,'VariableNames', varNames);

for ii=1:size(faces,1)
    facesIMDB.imageFilename(ii) = {[faces(ii).folder filesep faces(ii).name]};
    facesIMDB.face(ii) = {[1 1 32 32]};
end 

positiveInstances = facesIMDB;

%% VJ detector training

tic
trainCascadeObjectDetector("myFaceDetector.xml",positiveInstances,...
    negativeFolder, NegativeSamplesFactor = 2,...
    NumCascadeStages=10, FalseAlarmRate=0.01, TruePositiveRate=0.99, ...
    FeatureType='HOG'); 
toc

%% visualize the results 

detector = vision.CascadeObjectDetector('myFaceDetector.xml');
%detector = vision.CascadeObjectDetector();  % matlab detector

imgs = dir('./CaltechFaces/test_scenes/test_jpg/*.jpg');

for ii=1:size(imgs,1)
    img = imread([imgs(ii).folder filesep imgs(ii).name]);
    bbox = step(detector, img); % bbox will be an empty vector if no faces are detected, otherwise will be a matrix whose rows give us the bounding boxes

    detectedImg = insertObjectAnnotation(img,'rectangle',bbox,'face');
    detectedImg = imresize(detectedImg, 800/max(size(detectedImg)));

    figure(1), clf
    imshow(detectedImg)
    %waitforbuttonpress
end 

close all

%% compute our results

load('./CaltechFaces/test_scenes/GT.mat');

detector = vision.CascadeObjectDetector('myFaceDetector.xml');
%detector = vision.CascadeObjectDetector(); % matlab detector

imgs = dir('./CaltechFaces/test_scenes/test_jpg/*.jpg');


numImages = size(imgs, 1);
results = table('Size',[numImages 2],...
    'VariableTypes', {'cell','cell'},...
    'VariableNames',{'face','Scores'});

for ii=1:size(imgs,1)
    img = imread([imgs(ii).folder filesep imgs(ii).name]);
    bbox = step(detector, img);
    results.face{ii}=bbox;
    results.Scores{ii}=0.5+zeros(size(bbox,1),1);
end


%% Compute average precision

[ap, recall, precision] = evaluateDetectionPrecision(results, GT,0.2);
figure(2),clf
plot(recall, precision, 'g', 'LineWidth',2)
xlim([0 1])
ylim([0 1])
grid on
title(sprintf('Average Precision = %.2f',ap)) 
waitforbuttonpress


%% Test images with detected faces and groundtruths

load('./CaltechFaces/test_scenes/GT.mat');
imgs = dir('./CaltechFaces/test_scenes/test_jpg/*.jpg');
detector = vision.CascadeObjectDetector('myFaceDetector.xml');

for i=1:size(GT,1)

    img = imread([imgs(i).folder filesep imgs(i).name]);
    nFaces = numel(GT{i,:}{1})/4;                   % number of faces in the image
    bbox = step(detector, img);                      % box detected

    for ii=1:nFaces
        % extracting the coordinates of the 4 corners of each box
        positions = GT{i,:}{1}(ii,:);
        x1 = positions(1);
        x2 = positions(2);
        x3 = positions(3);
        x4 = positions(4);

        img = insertShape(img, 'Rectangle', [x1 x2 x3 x4], 'LineWidth', 1,...
              'Color', 'green'); % inserting the boxes around real faces 
    end  

    if numel(bbox)>0
        for jj = 1:size(bbox, 1)
            img = insertObjectAnnotation(img,'rectangle',bbox,'face'); % inserting detected boxes
        end
    end

    img = imresize(img, 800/max(size(img)));
    imshow(img);
    waitforbuttonpress
end