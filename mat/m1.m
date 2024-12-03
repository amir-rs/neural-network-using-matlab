% Load and prepare data
data = readtable('co2.csv', 'VariableNamingRule', 'preserve');

% Display column names
disp(data.Properties.VariableNames)

% Feature selection
X = [data.("Engine Size(L)"), data.Cylinders, ...
    data.("Fuel Consumption City (L/100 km)"), ...
    data.("Fuel Consumption Hwy (L/100 km)"), ...
    data.("Fuel Consumption Comb (L/100 km)"), ...
    data.("Fuel Consumption Comb (mpg)")];

% Target variable
Y = data.("CO2 Emissions(g/km)");

% Convert to double
X = double(X);
Y = double(Y);

% Analyze data range and detect outliers
figure;
histogram(Y, 30);
title('Distribution of CO2 Emissions');
xlabel('CO2 Emissions (g/km)');
ylabel('Frequency');

% Remove outliers using z-score threshold
z_scores = abs(zscore(Y));
outlier_idx = z_scores > 3;
X(outlier_idx, :) = [];
Y(outlier_idx, :) = [];

% Split data
cv = cvpartition(size(X,1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
X_test = X(test(cv), :);
Y_train = Y(training(cv), :);
Y_test = Y(test(cv), :);

% Normalize data
[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;

% Create and configure neural network
net = fitnet([128 64 32]);
net.trainFcn = 'trainscg';
net.performFcn = 'mse';
net.trainParam.epochs = 200;  % Increased epochs for better training
net.trainParam.goal = 1e-6;   % Lower error goal
net.trainParam.min_grad = 1e-7;
net.trainParam.max_fail = 10;
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0.1;

% Add regularization
net.performParam.regularization = 0.1;  % L2 regularization

% Train network
[net, tr] = train(net, X_train_norm', Y_train');

% Make predictions
Y_pred = net(X_test_norm')';

% Calculate regression metrics
mse = mean((Y_test - Y_pred).^2);
rmse = sqrt(mse);
mae = mean(abs(Y_test - Y_pred));
mape = mean(abs((Y_test - Y_pred) ./ Y_test)) * 100;
r2 = 1 - sum((Y_test - Y_pred).^2) / sum((Y_test - mean(Y_test)).^2);

% Display metrics
fprintf('MSE: %.2f\n', mse);
fprintf('RMSE: %.2f\n', rmse);
fprintf('MAE: %.2f\n', mae);
fprintf('MAPE: %.2f%%\n', mape);
fprintf('R2: %.4f\n', r2);

% Plot results
figure;
plot(Y_test, 'b');
hold on;
plot(Y_pred, 'r');
legend('Actual', 'Predicted');
title('CO2 Emissions: Actual vs Predicted');
xlabel('Sample');
ylabel('CO2 Emissions (g/km)');

% Plot regression
figure;
plotregression(Y_test, Y_pred, 'Regression');

% Error analysis
errors = Y_test - Y_pred;
figure;
histogram(errors, 20);
title('Error Histogram');
xlabel('Prediction Error');
ylabel('Frequency');

% Save model
save('co2_model_optimized.mat', 'net', 'mu', 'sigma');