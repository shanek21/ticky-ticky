function two_pendulums
    clear all
    close all
    clf
    
%% Parameters
    % System parameters
    m = 0.0605;  % mass of bob in kg
    M = 1.3280;  % mass of cart in kg
    weight = 0;  % added mass to the cart
    M = M + weight;
    mu = m/(M + 2*m); % Cart to bob mass ratio
    l = 0.327;   % pendulm length in m
    g = 9.80665; % acceleration due to gravity
    b = 0.02037; % friction coefficient clock
    B = .2;       % friction coefficient cart
    K = 0;       % cart spring rate
    
    % Set the clock parameters
    gamma = b*sqrt(l/4*g);  % Dimensionless parameter for clock
    period = sqrt(l/g);
    
    % Set the platform parameters
    Gamma = B*sqrt(l/4*g)/(M+2*m);     % Dimensionless parameter for clock on cart
    Omega = K/(M+2*m)*sqrt(g/l);       % Dimensionless poarameter for cart restoring force
    
%% Initial Conditions
    % Define the initial conditions (using original variables)
    theta1_0 = .3;
    theta2_0 = 0;
    thetaDot1_0 = 0;
    thetaDot2_0 = 0;
    y_0 = 0;
    ydot_0 = 0;

    % Convert to the sum and difference
    delta_0 = theta1_0 - theta2_0;
    deltadot_0 = thetaDot1_0 - thetaDot2_0;
    sigma_0 = theta1_0 + theta2_0;
    sigmadot_0 = thetaDot1_0 + thetaDot2_0;

    % pack them into an appropriate vector
    init = [delta_0;deltadot_0;sigma_0;sigmadot_0;y_0;ydot_0];

    % Define the start and end times
    Tstart=0;
    Tend=10000;
    tspan = [Tstart, Tend];

%% Solve and Unpack
    [T,M]=ode45(@clockodes,tspan,init);
    
    %Add dimensions to time
    T = T * period;
    
    % Get the angles and velocities
    theta1 = (M(:,1)+M(:,3))/2;
    theta2 = (M(:,3)-M(:,1))/2;
    cart = M(:,5) * l;
    thetaDot1 = (M(:,2)+M(:,4))/2;
    thetaDot2 = (M(:,4)-M(:,2))/2;

%% Plot
    figure(1)
    yyaxis left
    plot(T, theta1, '-r'), hold on
    plot(T, theta2, '-b')
    ylabel('Pendulum Angle (radians)')
    
    yyaxis right
    plot(T, cart * 100, 'k')
    ylabel('Cart Location (mm)')
    ylim([-5 5])
    
    xlabel('Time (s)')
    title('Antiphase with Angle Offset and Cart Damping')
    legend('Pendulum 1','Pendulum 2','Cart')
    xlim([0 60])

%% Derivatives Function
        function derivs = clockodes(t,W)
            delta = W(1);
            deltaDot = W(2);
            sigma = W(3);
            sigmaDot = W(4);
            Y = W(5);
            YDot = W(6);
   
            dY = YDot;
            ddelta = deltaDot;
            dsigma = sigmaDot;
            
            dYDot = (mu*(2*gamma*sigmaDot*abs(sigmaDot)+sigma)-2*Gamma*YDot-Omega^2*Y)/(1-2*mu) ;
            dsigmaDot = -2*gamma*sigmaDot*abs(sigmaDot)-sigma-2*dYDot;
            ddeltaDot = -2*gamma*deltaDot*abs(deltaDot)-delta;

            derivs = [ddelta; ddeltaDot; dsigma; dsigmaDot; dY; dYDot];
        end
    
%% Peak Timing
%     Mu = [];
%     Mass = [];
%     Frequency = [];
%     for M = 1:.2:10
%         Mass = [Mass M];
%         mu = m/(M + 2*m);
%         theta1_0 = pi/16;
%         theta2_0 = 0;
% 
%         % Convert to the sum and difference
%         delta_0 = theta1_0 - theta2_0;
%         deltadot_0 = thetaDot1_0 - thetaDot2_0;
%         sigma_0 = theta1_0 + theta2_0;
%         sigmadot_0 = thetaDot1_0 + thetaDot2_0;
% 
%         % pack them into an appropriate vector
%         init = [delta_0;deltadot_0;sigma_0;sigmadot_0;y_0;ydot_0];
% 
%         % Define the start and end times
%         Tstart=0;
%         Tend=1000;
%         tspan = [Tstart, Tend];
%         
%         [T,M]=ode45(@clockodes,tspan,init);
%     
%         %Add dimensions to time
%         T = T * period;
% 
%         % Get the angles
%         theta1 = (M(:,1)+M(:,3))/2;
%         
%         [pks1,locs] = findpeaks(theta1);
%         T_peaks = T([locs]);
%         [pks2,locs] = findpeaks(pks1);
%         T_shifts = T_peaks([locs]);
%         
%         frequency = T_shifts(1);
%         
%         Mu = [Mu mu];
%         Frequency = [Frequency frequency];
%     end
%     
%     figure(2)
%     plot(Mass, Frequency)
%     
%     xlabel(' Mu (m/(M + 2*m))')
%     ylabel('Time to Shift')
    
end