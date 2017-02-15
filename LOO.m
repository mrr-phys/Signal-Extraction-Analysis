%--------------------------------------------------------------------------
% Description: The following code imports the *.csv data files and 
% perform Leave-one-out cross-validation for an Ordinary Least Square (OLS), 
% Feasible Generalized Least square(FGLS), and Least Absolute Deviation (LAD) 
% In the end, it prints out a CSV file with the estimated coefficient \beta
% and the value that is removed and the the p-value 

%% Load the input data files in csv format
nof = 3; % number of input files

for i=3:nof
    filename=strcat('data_1_',int2str(i),'.csv');
    data=csvread(filename,1,0);
    % Get X, y
    [n,p]=size(data);
    x = data(:,1); y=data(:,2);
    X=[ones(n,1) x];
    
    %% Run the Leave-one-out Cross Validation 
    
    bHat_ols_vec = zeros( n , 2 );
    bHat_fgls_vec = zeros( n , 2 );
    bHat_lad_vec = zeros( n , 2 );
    
    pVal_ols = zeros( n , 1 );
    pVal_fgls = zeros( n , 1 );
    
    rmvPoint_index = zeros( n , 1 );
    rmvPoint_val = zeros( n , 1 );
    
    for k=1:n
        
        trainIdx=ones(n,1);
        trainIdx(k)=0;
        rmvPoint_index(k)=k;
        rmvPoint_val(k)=y(k);
        
        X_train = X(logical(trainIdx),:);
        y_train = y(logical(trainIdx));
 
        
        %% Ordinary Least Square (OLS)

        [bHat_ols,se_ols] = lscov(X_train,y_train);
            
        bHat_ols_vec(k,:) = bHat_ols;

        %Compute the p-value
        tstat= bHat_ols./se_ols;
        pVal_ols(k)=pdf('t',tstat(2,1),n);

  
        %% Feasible Generalized Least Square with heteroscedasticity consistent
    % covariance HC0, HC1, HC3, HC4
    
         [bHat_fgls,se_loo] = fgls(X_train(:,2),y_train,'InnovMdl', 'HC3'); 
         
         bHat_fgls_vec(k,:) = bHat_fgls;
         
         tstat= bHat_fgls./se_loo;
         pVal_fgls(k) = pdf('t',tstat(2,1),n);
         
         
         %% %Least Absolute Deviation (LAD)
          %  bHat_lad_vec(k,:) = lad(X_train,y_train);
         
        
    end
% Assign Output
Obs = rmvPoint_index;
rmv_Point = rmvPoint_val;

bHat1_ols = bHat_ols_vec(:,2);
bHat0_ols = bHat_ols_vec(:,1);

bHat1_fgls = bHat_fgls_vec(:,2);
bHat0_fgls = bHat_fgls_vec(:,1);

bHat1_lad = bHat_lad_vec(:,2);
bHat0_lad = bHat_lad_vec(:,1);

%T = {Obs,rmv_Point, bHat0_lad, bHat1_lad };

header = ['Obs' 'rmv_Point' 'pVal_ols' 'bHat0_ols' 'bHat1_ols' ...
    'pVal_fgls' 'bHat0_fgls' 'bHat1_fgls'];


T = {Obs, rmv_Point, pVal_ols, ...
    bHat0_ols,bHat1_ols, pVal_fgls,bHat0_fgls, bHat1_fgls};

dlmwrite('/Users/mohammad/Dropbox/voleon/VoleonAnalysis/LOO.csv',T,'delimiter',',','precision','%10.3f');




end