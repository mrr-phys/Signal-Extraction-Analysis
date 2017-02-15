%--------------------------------------------------------------------------
% Description: The following code imports the *.csv data files and fit an
% Ordinary Least Square (OLS), Feasible Generalized Least square(FGLS) and
% Least Absolute Deviation (LAD) to the model y=X*\beta + e in which
%\beta is the coeffcient to be predicted given input y and X. In the end
% it computes and plot the redisuals for the analysis of
% heteroscedasticity.
%--------------------------------------------------------------------------

clear; clc; 
    
%% Load the input data files in csv format
nof = 4; % number of input files

for i=4:nof
    filename=strcat('data_1_',int2str(i),'.csv');
    data=csvread(filename,1,0);
    % Get X, y
    [n,p]=size(data);
    x = data(:,1); y=data(:,2);
    X=[ones(n,1) x];

    

    % Comparing different models to fit the data
     %% Ordinary Least Square (OLS)

    [bHat_ols,se_ols] = lscov(X,y);
    yHat_ols = X*bHat_ols;

    %Compute the p-value
    tstat= bHat_ols./se_ols;
    pVal_ols=pdf('t',tstat,n);
    
    % Print p_value:
    disp(' ')
    disp(strcat('               data_', num2str(i)))
    disp(' ')
    disp('========================================================')
    disp('p-value for OLS model is:')
    disp(pVal_ols)
    disp('========================================================')
    disp(' ')
       
    %% Feasible Generalized Least Square with heteroscedasticity consistent
    % covariance HC0, HC1, HC3, HC4
    
    mdl = 'HC3';   
    [bHat_fgls ,se_fgls]=fgls(x,y,'InnovMdl',mdl);%
    yHat_fgls=X*bHat_fgls;
    
    %Compute the p-value
    tstat= bHat_fgls./se_fgls;
    pVal_fgls=pdf('t',tstat,n);
    
    disp(' ')
    disp('========================================================')
    disp(strcat('p-value for FGLS_', mdl))
    disp(pVal_fgls)
    disp('========================================================')
    disp(' ')



    %% Least Absolute Deviation (LAD) min_{\beta} ||y - X*\beta||_1 by employing 
    % linear programming solver linprog
    bHat_lad=lad(X,y);
    yHat_lad=X*bHat_lad;
    
 % MAKEPLOTS ---------------------------------------------------------------
    %% Compute the residuals from the OLS fit and plot
    [xsorted,I]=sort(x);
    z = y - yHat_ols;
    %[h,pValue] = archtest(z);
   % disp(h) 
   % disp(pValue)
    z_I = z(I);
    
    figure(1)
    hold all
    % Plot the residual in figure 1
    plot(xsorted,smooth(z_I.^2,10),'LineWidth',2)
    xlabel('x')
    ylabel('squared residual')
    title('Squared residuals vs. x')

    Legend=cell(nof,1);
    for l=1:nof
      Legend{l}=strcat('data', num2str(l));
    end
    legend(Legend, 'Location','northwest')
    drawnow
    pause(1)
    %% Compute the residuals from the OLS fit vs. predicted
    
    figure
    hold on
    % Plot the residual in figure 1
    scatter(yHat_ols,z)
    xlabel('predicted')
    ylabel('residual')
    title('residuals vs. predicted')

    legend(strcat('data_', num2str(i)))
    drawnow
    pause(1)   
    %%%%%%%%%
    %% Plot the AutoCorr and Partial Corr for sign of correlation
    figure
    subplot(2,1,1);
    autocorr(z);
    legend(strcat('data_', num2str(i)))
    subplot(2,1,2);
    parcorr(z);
    drawnow
    pause(1)
    hold on
    %% Scatter plot and the fitted line for OLS,LAD and FGLS models 
    figure
    scatter(x,y)
    xlabel('x')
    ylabel('y')
    title(strcat('data_', num2str(i)))
    pause(.5)
    hold on
    plot(x,yHat_ols , x, yHat_fgls,x,yHat_lad,'LineWidth',1)
    legend('Scatter Plot','OLS', strcat('FGLS\_', mdl),'LAD', 'Location', 'northwest')
    drawnow
    pause(1)
    %%%%%%%%%%%%
end

