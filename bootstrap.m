%--------------------------------------------------------------------------
% Description: The following code imports the *.csv data files and 
% perform pair bootstraping on 10000 resampling for
% an Ordinary Least Square (OLS), Feasible Generalized Least square(FGLS)
% and Least Absolute Deviation (LAD).
%--------------------------------------------------------------------------
%% Load the input data files in csv format
nof = 5; % number of input files

bootstrap_SD = zeros(2,nof);  % avrg_rmse array for 5 datasets

for i=1:nof
    filename=strcat('data_1_',int2str(i),'.csv');
    data=csvread(filename,1,0);
    % Get X, y
    [n,p]=size(data);
    x = data(:,1); y=data(:,2);
    X=[ones(n,1) x];
        
 %% Bootstrap resampling pair (x*,y*)   
    bHat = fgls(x, y,'InnovMdl','HC3','numIter',3);%X\y;%lad(X,y);
    
    B = 10000; 
    % Initialize beta_star
    bStar = zeros( B , 2 );
    % Loop Over B bootstraps
    for b=1:B
        %Uniform random numbers over 1...n
        u = ceil(n*rand(n,1));
        
        % X-star sample simulation
        y_star = y(u);
        X_star = X(u,:);
        
        
        % Mean of X star
        bStar(b,:) =  fgls(X_star(:,2), y_star,'InnovMdl','HC3','numIter',3);%X_star\y_star;%%lad(X_star,y_star);
    
    end

    beta_err=bsxfun(@minus,bStar,bHat');
    % Standard Deviation (SD) estimate
    bootstrap_SD(:,i) = diag(beta_err'*beta_err)/B;
end


