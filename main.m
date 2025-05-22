clc;
clear;

% === Setup paths ===
currentFolder = fileparts(mfilename('fullpath'));
datasetPath = fullfile(currentFolder, 'ASL_Dataset');
modelFile = 'ASL_ABC_CNN.mat';

% === Preprocessing Function ===
preprocessImage = @(img) imresize(rgb2gray(img), [128, 128]);

% === Load or Train Model ===
if isfile(modelFile)
    disp('✔️ Loading the saved model...');
    load(modelFile, 'net');
else
    disp('🚀 No saved model found. Starting training...');
    [trainData, testData, numClasses] = preprocessASL(datasetPath);
    net = trainASLModel(trainData, testData, numClasses);
    save(modelFile, 'net');
    disp('✅ Model training complete and saved.');
end

% === Mode Selection ===
choice = input(['\nChoose Mode:\n' ...
                '1 - Random Image from Testing\n' ...
                '2 - Webcam Snapshot\n' ...
                '3 - Random from Each Class (Validation)\n' ...
                '4 - Live Video Prediction\n' ...
                'Your choice: ']);

% === MODE 1: Random image from Testing ===
if choice == 1
    testFolder = fullfile(datasetPath, 'Testing');
    classFolders = getSubfolders(testFolder);

    randClass = classFolders{randi(numel(classFolders))};
    imgFiles = dir(fullfile(testFolder, randClass, '*.jpg'));

    if isempty(imgFiles)
        error('❌ No images found in folder: %s', randClass);
    end

    randImg = imgFiles(randi(numel(imgFiles)));
    imgPath = fullfile(testFolder, randClass, randImg.name);
    img = imread(imgPath);

    imgProcessed = preprocessImage(img);
    predLabel = classify(net, imgProcessed);

    figure; imshow(imgProcessed);
    title(['Prediction: ', char(predLabel), ' | Ground Truth: ', randClass]);

% === MODE 2: Webcam Snapshot Prediction ===
elseif choice == 2
    disp('📸 Capturing image from webcam...');
    vid = videoinput('winvideo', 1);
    set(vid, 'ReturnedColorSpace', 'RGB');

    preview(vid); pause(3);  % Stabilize
    img = getsnapshot(vid);
    closepreview(vid); clear vid;

    imgProcessed = preprocessImage(img);
    predLabel = classify(net, imgProcessed);

    figure; imshow(imgProcessed);
    title(['Webcam Prediction: ', char(predLabel)]);

% === MODE 3: Random Image from Each Class ===
elseif choice == 3
    testFolder = fullfile(datasetPath, 'Testing');
    classFolders = getSubfolders(testFolder);

    if numel(classFolders) ~= 4
        error('❌ Expected exactly 4 class folders. Found %d.', numel(classFolders));
    end

    figure('Name', 'Validation: One Random Image per Class');

    for i = 1:4
        folderPath = fullfile(testFolder, classFolders{i});
        imgFiles = dir(fullfile(folderPath, '*.jpg'));
        if isempty(imgFiles)
            warning('⚠️ No images found in %s', classFolders{i});
            continue;
        end

        randImg = imgFiles(randi(numel(imgFiles)));
        imgPath = fullfile(folderPath, randImg.name);
        img = imread(imgPath);

        imgProcessed = preprocessImage(img);
        predLabel = classify(net, imgProcessed);

        subplot(1, 4, i);
        imshow(imgProcessed);
        title({['Predicted: ', char(predLabel)], ['Actual: ', classFolders{i}]});
    end

% === MODE 4: Live Real-Time ASL Detection ===
elseif choice == 4
    disp('📹 Starting live ASL prediction. Close the figure to stop.');
    vid = videoinput('winvideo', 1, 'YUY2_640x480');
    set(vid, 'ReturnedColorSpace', 'RGB');
    vid.FrameGrabInterval = 5;

    figure('Name', 'Live ASL Prediction');

    while ishandle(gcf)
        frame = getsnapshot(vid);
        imgProcessed = preprocessImage(frame);
        predLabel = classify(net, imgProcessed);

        imshow(frame);
        title(['Live Prediction: ', char(predLabel)], 'FontSize', 16);
        pause(0.1);
    end
    clear vid;

else
    disp('❌ Invalid choice. Please select 1, 2, 3 or 4.');
end

% === Helper Function to Get Folder Names ===
function folders = getSubfolders(parentDir)
    info = dir(parentDir);
    folders = {info([info.isdir] & ~ismember({info.name}, {'.', '..'})).name};
end
