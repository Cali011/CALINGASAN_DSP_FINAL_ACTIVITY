clc;
clear;

% Automatically get the current script's directory
currentFolder = fileparts(mfilename('fullpath'));

% Construct the full dataset path based on the script's location
datasetPath = fullfile(currentFolder, 'ASL_Dataset');
modelFile = 'ASL_ABC_CNN.mat';

% Check if the model file exists
if isfile(modelFile)
    disp('Loading the saved model...');
    load(modelFile, 'net');
else
    % Step 1: Preprocess the data
    disp('No saved model found. Starting training...');
    [trainData, testData, numClasses] = preprocessASL(datasetPath);

    % Step 2: Train and save the model
    net = trainASLModel(trainData, testData, numClasses);
end

% Step 3: Predict using saved image or webcam
choice = input('Choose mode: 1 - Random Image from Testing, 2 - Webcam:  3 - Validation: ');

if choice == 1
    % Random image mode from the Testing folder
    disp('Selecting a random image from the Testing folder...');
    
    % Get the path to the Testing folder
    testFolder = fullfile(datasetPath, 'Testing');
    
    % Get a list of all files in the Testing folder
    classes = dir(testFolder); 
    classes = classes([classes.isdir]);  % Filter out non-directory files
    classes = classes(~ismember({classes.name}, {'.', '..'}));  % Exclude '.' and '..'
    
    % Randomly select a class folder
    randClassIdx = randi(numel(classes));
    randClassFolder = fullfile(testFolder, classes(randClassIdx).name);
    
    % Get all image files in the selected class folder
    imgFiles = dir(fullfile(randClassFolder, '*.jpeg'));  % Change extension if needed
    if isempty(imgFiles)
        disp('No images found in the selected folder.');
        return;
    end
    
    % Randomly select an image file
    randImgIdx = randi(numel(imgFiles));
    imgPath = fullfile(randClassFolder, imgFiles(randImgIdx).name);
    
    % Predict the random image
    predictASL(net, 'image', imgPath);

elseif choice == 2
    % Webcam mode - capture an image using videoinput
    disp('Capturing image from webcam...');
    
    % Initialize video input (webcam)
    vid = videoinput('winvideo', 1);
    set(vid, 'ReturnedColorSpace', 'RGB');
    
    % Preview the video
    preview(vid);
    pause(5); % Let the preview stabilize for 5 seconds

    % Capture a snapshot
    img1 = getsnapshot(vid);
    
    % Close preview window
    closepreview(vid);

    % Resize the captured image to 64x64
    img1 = imresize(rgb2gray(img1), [128, 128]);

    % Predict using the captured image
    predLabel = classify(net, img1);

    % Display the captured image and prediction result
    imshow(img1);
    title(['Prediction: ', char(predLabel)]);
    
    % Clear the webcam object
    clear vid;

elseif choice == 3
    % Random image mode from each folder in the Testing directory
    disp('Selecting a random image from each folder in the Testing directory...');

    % Get the path to the Testing folder
    testFolder = fullfile(datasetPath, 'Testing');
    
    % Get a list of all subfolders (directories only)
    classes = dir(testFolder); 
    classes = classes([classes.isdir]);  % Filter out non-directory files
    classes = classes(~ismember({classes.name}, {'.', '..'}));  % Exclude '.' and '..'
    
    % Check if the number of folders is exactly 5
    if numel(classes) ~= 26
        disp('Error: The Testing folder must contain exactly 26 subfolders.');
        return;
    end

    % Create a single figure for displaying all images
    figure('Name', 'Randomly Selected Images from Each Folder', 'NumberTitle', 'off');

    % Loop through each folder and display the image
    for i = 1:numel(classes)
        % Get the path of the current class folder
        classFolder = fullfile(testFolder, classes(i).name);

        % Get all image files in the current class folder
        imgFiles = dir(fullfile(classFolder, '*.jpeg'));  % Change extension if needed

        % Check if any images were found in the current folder
        if isempty(imgFiles)
            disp(['No images found in folder: ', classes(i).name]);
            continue;
        end
        
        % Randomly select an image file from the current folder
        randImgIdx = randi(numel(imgFiles));
        imgPath = fullfile(classFolder, imgFiles(randImgIdx).name);
        
        % Display the selected image path
        disp(['Randomly selected image from folder ', classes(i).name, ': ', imgPath]);

        % Predict the random image
        img = imread(imgPath);
        img = imresize(rgb2gray(img), [128, 128]);  % Updated to 128x128
        predLabel = classify(net, img);
        
        % Display the image in the subplot
        subplot(6, 5, i);  % Arrange in a single row with 5 columns
        imshow(img);
        title(['Predicted: ', char(predLabel), ' (', classes(i).name, ')']);
    end
 
else
    disp('Invalid choice. Please select 1 for random image or 2 for webcam.');
end