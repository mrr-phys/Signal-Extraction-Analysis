%--------------------------------------------------------------------------
% Description: The following code imports the *.csv data files and 
% perform k-fold cross-validation for an Ordinary Least Square (OLS), 
% Feasible Generalized Least square(FGLS), and Least Absolute Deviation (LAD) 
% In the end, it computes the Root Mean Squared Error for each model.
%--------------------------------------------------------------------------

%% Load the input data files in csv format
nof = 5; % number of input files

avrg_rmse = zeros(7,nof);  % avrg_rmse array for 5 datasets
for i=1:nof
    filename=strcat('data_1_',int2str(i),'.csv');
    data=csvread(filename,1,0);
    % Get X, y
    [n,p]=size(data);
    x = data(:,1); y=data(:,2);
    X=[ones(n,1) x];
 
    %% Run the 5-fold Cross Validation P=1000 times 
    
    K=5;
    P = 1000;
 
    iter=1;
    
 
    mse_fgls_hc0 = zeros( P*K , 1 );
    mse_fgls_hc1 = zeros( P*K , 1 );    
    mse_fgls_hc2 = zeros( P*K , 1 );       
    mse_fgls_hc3 = zeros( P*K , 1 ); 
    mse_fgls_hc4 = zeros( P*K , 1 ); 
    
    mse_ols = zeros( P*K , 1 );
    mse_lad = zeros( P*K , 1 );
    
    for j=1:P
        
        % partition data into K folds
        cv = cvpartition(numel(y), 'kfold',K);

        for k=1:K
            
            
            % training/testing indices for this fold
            trainIdx = cv.training(k);
            testIdx = cv.test(k);       
           
            % X and y training  
            X_train = X(trainIdx,:);
            y_train = y(trainIdx);
           
            % testing X, and y
            X_test = X(testIdx,:);
            y_test = y(testIdx);

            %Feasible Generalized Least Square function with heteroscedasticity consistent
            % covariance HC = HC0, HC1, HC3, HC4
            
            bHat_fgls_hc0 = fgls(X_train(:,2), y_train,'InnovMdl','HC0','numIter',3);
            yHat_fgls_hc0 = X_test*bHat_fgls_hc0;
            
            bHat_fgls_hc1 = fgls(X_train(:,2), y_train,'InnovMdl','HC1','numIter',3);
            yHat_fgls_hc1 = X_test*bHat_fgls_hc1;
            
            bHat_fgls_hc2 = fgls(X_train(:,2), y_train,'InnovMdl','HC2','numIter',3);
            yHat_fgls_hc2 = X_test*bHat_fgls_hc2;
            
            bHat_fgls_hc3 = fgls(X_train(:,2), y_train,'InnovMdl','HC3', 'numIter',3);
            yHat_fgls_hc3 = X_test*bHat_fgls_hc3;
            
            bHat_fgls_hc4 = fgls(X_train(:,2), y_train,'InnovMdl','HC4','numIter',3);
            yHat_fgls_hc4 = X_test*bHat_fgls_hc4;
    
            % Ordinary Least Square (OLS)
            bHat_ols = lscov(X_train,y_train);
            yHat_ols = X_test*bHat_ols;
          
            %Least Absolute Deviation (LAD)
            bHat_lad = lad(X_train,y_train);
            yHat_lad = X_test*bHat_lad;
            
 
            % compute mean squared error
            mse_fgls_hc0(iter) = mean((y_test - yHat_fgls_hc0).^2);
            mse_fgls_hc1(iter) = mean((y_test - yHat_fgls_hc1).^2);            
            mse_fgls_hc2(iter) = mean((y_test - yHat_fgls_hc2).^2);            
            mse_fgls_hc3(iter) = mean((y_test - yHat_fgls_hc3).^2);
            mse_fgls_hc4(iter) = mean((y_test - yHat_fgls_hc4).^2);            
            
            mse_ols(iter) = mean((y_test - yHat_ols).^2);
            
            mse_lad(iter) = mean((y_test - yHat_lad).^2);

            iter =iter + 1;
        end
        
    end
   

  
  %%  average RMSE across k-folds for input data file i
  
    avrg_rmse(2,i) = mean(sqrt(mse_fgls_hc0));
    avrg_rmse(3,i) = mean(sqrt(mse_fgls_hc1));    
    avrg_rmse(4,i) = mean(sqrt(mse_fgls_hc2));
    avrg_rmse(5,i) = mean(sqrt(mse_fgls_hc3));    
    avrg_rmse(6,i) = mean(sqrt(mse_fgls_hc4));    
  
   
    avrg_rmse(1,i) = mean(sqrt(mse_ols));  
    avrg_rmse(7,i) = mean(sqrt(mse_lad));
    
    csvwrite('/Users/mohammad/Dropbox/voleon/VoleonAnalysis/CV.csv',avrg_rmse)
 




end