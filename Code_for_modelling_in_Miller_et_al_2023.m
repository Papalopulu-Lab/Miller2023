% The following is the code used for the data analysis used for results in
% Miller et. al.

clear all;
clc;
close all

%% Parameters and preallocation

h=0.1; % Step Size

% Set values for time delays
time_delay_N_to_X = 10;
time_delay_N_to_Y = 10;
time_delay_X_to_Y = 20;

% time delay vector
time_delay = [time_delay_N_to_X,time_delay_N_to_Y,time_delay_X_to_Y];

% Number of interations
iter=250+max(time_delay); 

% Create the history time vector 
pre_t = 0-max(time_delay) : h : 0-1;

% Create time vector from t=0 onwards
t=0:h:h*iter; % Time vector

%Label names
expt_name = {'PM','WT'};
%expt_name = {'WT1','WT2'};

%colours for plots
Line_type = {'k','r--'};

% Create arrays for the three components of the model
X = cell(1,numel(expt_name));
Y = cell(1,numel(expt_name));
N = cell(1,numel(expt_name));

% History arrays for the three compenents of the model
X_pre = cell(1,numel(expt_name));
Y_pre = cell(1,numel(expt_name));
N_pre = cell(1,numel(expt_name));

% Final panel of figure preallocation

Prob_of_diff = cell(1,length(N));

% Set mathematical model parameters
alpha_X = 1; % X production scalar
mu_X = 0.1; % X degradation scalar

alpha_Y = 1; % Y production scalar
mu_Y = 1; % Y degradation scalar

%%

% Since the same script is used for the modelling in each olution image in
% Figure 5. The following four sections contain different inputs for Ngn3.

% - Phosphomutant vs Wild tpye NGN3 dynamcs.

% - Two sine waves of NGN3 dynamics with changing fold change but equal 
% mean leavels.

% - Two sine waves of NGN3 dynamcs with changing mean levels but the smae
% fold change.

% - Two sine waves of NGN3 dynamics but with changing peak heights, one
% gradually increasing and one gradually decreasing.


% Only one of the following four sections should be readable code at any
% given time.


%% PM vs WT

% fold change scale factors for periodicity and amplitude

Omega_sf = 1.15;
Amp_sf = 1.3;

Amp(1) = 1*Amp_sf;
Amp(2) = 1;

Omega(1) = 1*Omega_sf;
Omega(2) = 1;

% History

History_of_N(1) = 0;
History_of_N(2) = 0;
History_of_X(1) = 1;
History_of_X(2) = 1;
History_of_Y(1) = 0;
History_of_Y(2) = 0;

% Assigning NGN3 dynamics 

% PM

N{1} = Amp(1)*sin((1/Omega(1))*t-pi/2)+Amp(1);
[~,Npm_trough_locs] = findpeaks(-N{1});
N{1}(Npm_trough_locs(1):end) = 0; %PM

%WT

N{2} = Amp(2)*sin((1/Omega(2))*t-pi/2)+Amp(2);
[~,Nwt_trough_locs] = findpeaks(-N{2});
N{2}(Nwt_trough_locs(3):end) = 0;

%% Different fold change same levels

% Amp(1) = 5;
% Amp(2) = 1;
% 
% Omega(1) = 1;
% Omega(2) = 1;
%
% % Assigning NGN3 dynamics 
%
% N{1} = Amp(1)*sin((1/Omega(1))*t-3*pi/2)+Amp(1)+Amp(2);
% N{2} = Amp(2)*sin((1/Omega(2))*t-3*pi/2)+Amp(2)+Amp(1);
%
% History_of_N(1) = max(N{1});
% History_of_N(2) = max(N{2});
%
% History_of_X(1) = 1;
% History_of_X(2) = 1;
% History_of_Y(1) = 0;
% History_of_Y(2) = 0;

%% Different levels same fold change

% fold_change = 2;
% 
% Omega = 1; % Periodicity
% 
% Abs_amplitude_1 = fold_change-1;
% Abs_amplitude_2 = Abs_amplitude_1*fold_change;
% 
% Amp = [Abs_amplitude_1/2,Abs_amplitude_2/2]; % Amplitude
% 
% Ngn_1 = 1;
% Ngn_2 = Ngn_1*fold_change;
% 
% Initial_point_of_ngn = [Ngn_1,Ngn_2];
% 
% % History
% 
% History_of_N(1) = Initial_point_of_ngn(1);
% History_of_N(2) = Initial_point_of_ngn(2);
% 
% % Assigning NGN3 dynamics 
% 
% N{1} = Amp(1)*sin((1/Omega)*t-pi/2)+Amp(1)+History_of_N(1);
% N{2} = Amp(2)*sin((1/Omega)*t-pi/2)+Amp(2)+History_of_N(2);
% 
% % History
% 
% History_of_X(1) = 1;
% History_of_X(2) = 2;
% History_of_Y(1) = 0;
% History_of_Y(2) = 0;

%% Different fold changes over time

% Differing heights of peak height was acvhieved by stitching together
% three different single-period, trough to trough sine waves.  
%
%
% % Time vectors
%
% x1 = 0:round(length(t)/4);
% x2 = round(length(t)/4)+1:2*round(length(t)/4);
% x3 = 2*round(length(t)/4)+1:3*round(length(t)/4);
% x4 = 3*round(length(t)/4)+1:4*round(length(t)/4);
%
%
% % Amplitudes
%
% A1 = 1;
% A2 = 2;
% A3 = 3;
%
% % the three sine wave functions followed by a vector of zeros
%
% y1 = A1*sin((2*pi/round(length(t)/4))*x1-pi/2)+A1;
% y2 = A2*sin((2*pi/round(length(t)/4))*x2-pi/2)+A2;
% y3 = A3*sin((2*pi/round(length(t)/4))*x3-pi/2)+A3;
% y4 = zeros(1,length(x4));
%
% % Assigning NGN3 dynamics 
%
% N{1} = [y1,y2,y3,y4];
% N{2} = [y3,y2,y1,y4];
%
% % History
%
% History_of_N(1) = 0;
% History_of_N(2) = 0;
% History_of_X(1) = 1;
% History_of_X(2) = 1;
% History_of_Y(1) = 0;
% History_of_Y(2) = 0;


%% 
% 
% The following code loops through the two lines within 
% the case of NGN3 dynamics we have selected. Solving the coupled ODEs to
% find simulations for X and Y via a 4th order Runge-Kutta numerical
% method.

for dynamics_type = 1:length(N)

    % Initial conditionsC

    X{dynamics_type}=zeros(1,length(t));
    X{dynamics_type}(1)=History_of_X(dynamics_type) ;

    Y{dynamics_type}=zeros(1,length(t));
    Y{dynamics_type}(1)=History_of_Y(dynamics_type) ;

    % History

    N_pre{dynamics_type}(1:max(time_delay)) = History_of_N(dynamics_type)  ;
    X_pre{dynamics_type}(1:max(time_delay)) = History_of_X(dynamics_type)  ;
    Y_pre{dynamics_type}(1:max(time_delay)) = History_of_Y(dynamics_type)  ;

    N{dynamics_type} = [N_pre{dynamics_type},N{dynamics_type}];
    X{dynamics_type} = [X_pre{dynamics_type},X{dynamics_type}];
    Y{dynamics_type} = [Y_pre{dynamics_type},Y{dynamics_type}];

    for j=max(time_delay)+1:iter

        % Preallocating the delayed variables

        N_to_X_delayed = N{dynamics_type}(j-time_delay_N_to_X);
        N_to_Y_delayed = N{dynamics_type}(j-time_delay_N_to_Y);
        X_to_Y_delayed = X{dynamics_type}(j-time_delay_X_to_Y);

        % X numerical scheme

        k1_X=our_function_X_RK(X{dynamics_type}(j),            N_to_X_delayed,alpha_X,mu_X);
        k2_X=our_function_X_RK(X{dynamics_type}(j)+0.5*h*k1_X, N_to_X_delayed,alpha_X,mu_X);
        k3_X=our_function_X_RK(X{dynamics_type}(j)+0.5*h*k2_X, N_to_X_delayed,alpha_X,mu_X);
        k4_X=our_function_X_RK(X{dynamics_type}(j)+h*k3_X,     N_to_X_delayed,alpha_X,mu_X);

        X{dynamics_type}(j+1)=X{dynamics_type}(j)+(h/6)*(k1_X+2*k2_X+2*k3_X+k4_X);

        % Y numerical scheme

        k1_Y=our_function_Y_RK(Y{dynamics_type}(j),            N_to_Y_delayed,X_to_Y_delayed,alpha_Y,mu_Y);
        k2_Y=our_function_Y_RK(Y{dynamics_type}(j)+0.5*h*k1_Y, N_to_Y_delayed,X_to_Y_delayed,alpha_Y,mu_Y);
        k3_Y=our_function_Y_RK(Y{dynamics_type}(j)+0.5*h*k2_Y, N_to_Y_delayed,X_to_Y_delayed,alpha_Y,mu_Y);
        k4_Y=our_function_Y_RK(Y{dynamics_type}(j)+h*k3_Y,     N_to_Y_delayed,X_to_Y_delayed,alpha_Y,mu_Y);

        Y{dynamics_type}(j+1)=Y{dynamics_type}(j)+(h/6)*(k1_Y+2*k2_Y+2*k3_Y+k4_Y);

    end


    %% Plotting figures

    % Figure 1 does not contain probaility of differntiation as a fourth
    % subplot panel (and thus labels are changed to aesthetically account for that)
    % whereas figure 2 does.

    % figure(1)
    % subplot(3,1,1)
    % plot(N{dynamics_type},Line_type{dynamics_type},'LineWidth',2)
    % xlim([max(time_delay),iter])
    % xlabel('Time')
    % ylabel('Solution')
    % title('Ngn3')
    % hold on
    % subplot(3,1,2)
    % plot(X{dynamics_type},Line_type{dynamics_type},'LineWidth',2)
    % xlim([0,iter])
    % %xlim([0,40])
    % xlabel('Time')
    % ylabel('Solution')
    % title('X')
    % hold on
    % subplot(3,1,3)
    % plot(Y{dynamics_type},Line_type{dynamics_type},'LineWidth',2)
    % xlabel('Time')
    % ylabel('Solution')
    % title('Y')
    % hold on
    % xlim([0,iter])
    % %xlim([0,40])
    % legend(expt_name)

    figure(2)
    subplot(4,1,1)
    plot(N{dynamics_type}(max(time_delay)+1:iter),Line_type{dynamics_type},'LineWidth',2)
    xticks([])
    ylabel({'Ngn3 expression';'simulation'})
    hold on
    subplot(4,1,2)
    plot(X{dynamics_type}(max(time_delay)+1:iter)-1,Line_type{dynamics_type},'LineWidth',2)
    xticks([])
    ylabel('X')
    hold on
    subplot(4,1,3)
    plot(Y{dynamics_type}(max(time_delay)+1:iter),Line_type{dynamics_type},'LineWidth',2)
    xticks([])
    xlabel('Time')
    hold on
    legend(expt_name)
    subplot(4,1,4)
    % Define our "probability of differentiation" as the average of solutions X, Y
    % divided by their respective cumulative sums.
    Prob_of_diff{dynamics_type} = (X{dynamics_type}(max(time_delay)+1:iter)./sum(X{dynamics_type}(max(time_delay)+1:iter))+...
        Y{dynamics_type}(max(time_delay)+1:iter)./(sum(Y{dynamics_type}(max(time_delay)+1:iter))))./2;
    plot(Prob_of_diff{dynamics_type},Line_type{dynamics_type},'LineWidth',2)
    xlabel('Time')
    ylabel({'Cumulative Probability';'to Differentiation'})
    hold on

end


%% Here is where we assign the functions for the ODEs for X and Y

function fx = our_function_X_RK(X,N,alpha_X,mu_X)

fx = alpha_X*N - mu_X*X;

end

function fy = our_function_Y_RK(Y,N,X,alpha_Y,mu_Y)

fy = alpha_Y*N/X - mu_Y*Y;

end








