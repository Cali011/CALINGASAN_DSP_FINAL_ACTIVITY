function net = trainASLModel(trainData, testData, numClasses)
    % Define the CNN architecture
    layers = [
        imageInputLayer([128 128 1], 'Normalization', 'none')  % Updated to 128x128
        convolution2dLayer(3, 16, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)

        convolution2dLayer(3, 32, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)

        convolution2dLayer(3, 64, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(numClasses)    % Dynamically set the number of classes
        softmaxLayer
        classificationLayer];

    % Training options
    options = trainingOptions('adam', ...
        'InitialLearnRate', 0.001, ...
        'MaxEpochs', 20, ...
        'MiniBatchSize', 32, ...
        'ValidationData', testData, ...
        'ValidationFrequency', 10, ...
        'Verbose', true, ...
        'Plots', 'training-progress');

    % Train the model
    net = trainNetwork(trainData, layers, options);

    % Save the trained model
    save('ASL_ABC_CNN.mat', 'net');
end
