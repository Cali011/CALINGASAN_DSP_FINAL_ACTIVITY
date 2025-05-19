function predictASL(net, mode, input)
    if strcmp(mode, 'image')
        % Load a single image for prediction
        img = imread(input);
        img = imresize(rgb2gray(img), [128, 128]);  % Updated to 128x128

        % Predict the label
        predLabel = classify(net, img);

        % Display result
        imshow(img);
        title(['Predicted: ', char(predLabel)]);
        
    elseif strcmp(mode, 'webcam')
        % Real-time prediction using webcam
        cam = webcam;
        disp('Press Ctrl+C to stop...');
        while true
            % Capture frame
            img = snapshot(cam);
            imgGray = imresize(rgb2gray(img), [128, 128]);  % Updated to 128x128

            % Classify the frame
            predLabel = classify(net, imgGray);

            % Display the frame with prediction
            imshow(img);
            title(['Prediction: ', char(predLabel)]);
            drawnow;
        end
        clear cam;
    else
        error('Invalid mode. Use "image" or "webcam".');
    end
end
