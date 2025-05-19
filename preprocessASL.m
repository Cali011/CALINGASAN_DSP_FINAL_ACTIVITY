function [trainData, testData, numClasses] = preprocessASL(datasetPath)
    % Load the training dataset
    trainPath = fullfile(datasetPath, 'Training');
    trainData = imageDatastore(trainPath, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');

    % Load the testing dataset
    testPath = fullfile(datasetPath, 'Testing');
    testData = imageDatastore(testPath, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');

    % Display the number of images per category
    disp('Training set image counts:');
    disp(countEachLabel(trainData));
    disp('Testing set image counts:');
    disp(countEachLabel(testData));

    % Resize images to [128x128] and convert to grayscale
    trainData.ReadFcn = @(loc) imresize(rgb2gray(imread(loc)), [128, 128]);
    testData.ReadFcn = @(loc) imresize(rgb2gray(imread(loc)), [128, 128]);

    % Get the number of unique classes
    numClasses = numel(unique(trainData.Labels));
end