function prediction = predict_co2(input_data)
    % Load saved model and preprocessing parameters
    load('co2_model.mat', 'net', 'mu', 'sigma');
    
    % تبدیل به double
    input_data = double(input_data);
    
    % Normalize input
    input_norm = (input_data - mu) ./ sigma;
    
    % Make prediction
    prediction = net(input_norm')';
end