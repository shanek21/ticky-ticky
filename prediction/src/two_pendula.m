function two_pendulums
    clear all
    close all
    
    % System parameters
    m = 0.0605;  % mass of bob in kg
    M = 1.3280;  % mass of cart in kg
    l = 0.327;   % pendulm length in m
    g = 9.80665; % acceleration due to gravity
    b = 0.00186; % friction coefficient clock
    B = 0;       % friction coefficient cart
    K = 0;       % cart spring rate
    
    % Set the clock parameters
    Phi = 0.1;              % Escapement thing
    gamma = b*sqrt(l/4*g);  % Dimensionless parameter for clock
    c = 1;                  % Escapement thing
    e = 0.05;               % Escapement thing
    
    % Set the platform parameters
    Gamma = B*sqrt(l/4*g)/(M+2*m);     % Dimensionless parameter for clock on cart
    Omega = K/(M+2*m)*sqrt(g/l);       % Dimensionless poarameter for cart restoring force
    mu = m/(M + 2*m);                  % Cart to bob mass ratio
    
    l = 0.32;
    g = 9.8;
    m = .0605;
    M = 1.328;
    mu = m/(M + 2*m);
    b = .00186;
    gamma = b*sqrt(l/(4*g));
    gamma = b;
    B = .0;
    Gamma = B*sqrt(l/(4*g))*(M+2*m);
    period = sqrt(l/g);
   
    
    % Define the initial conditions (using original variables)
    theta1 = pi/4;
    thetaDot1 = 0;
    theta2 = 0;
    thetaDot2 = 0;
    y = 0;
    ydot = 0;

    % Convert to the sum and difference
    delta = theta1 - theta2;
    deltadot = thetaDot1 - thetaDot2;
    sigma = theta1 + theta2;
    sigmadot = thetaDot1 + thetaDot2;

    % pack them into an appropriate vector
    init = [delta;deltadot;sigma;sigmadot;y;ydot];

    % Define the start and end times
    Tstart=0;
    Tend=2000;
    
    tspan = [Tstart, Tend];


    [T,M]=ode45(@clockodes,tspan,init);
    
    % Get the angles and velocities
    theta1 = (M(:,1)+M(:,3))/2;
    theta2 = (M(:,3)-M(:,1))/2;
    cart = M(:,5);
    thetaDot1 = (M(:,2)+M(:,4))/2;
    thetaDot2 = (M(:,4)-M(:,2))/2;

    plot(T*period, theta1, 'r'), hold on
    plot(T*period, theta2, 'b')
    plot(T*period, cart, 'k')
    
    legend('Pendulum 1','Pendulum 2','Cart')




        function derivs = clockodes(t,W)
            delta = W(1);
            deltaDot = W(2);
            sigma = W(3);
            sigmaDot = W(4);
            Y = W(5);
            YDot = W(6);
   
            dY =YDot;
            ddelta = deltaDot;
            dsigma = sigmaDot;
            
            dYDot = (mu*(2*gamma*deltaDot*abs(deltaDot)+delta)-2*Gamma*YDot-Omega^2*Y)/(1-2*mu) ;
            ddeltaDot = -2*gamma*deltaDot*abs(deltaDot)-delta-2*dYDot;
            dsigmaDot = -2*gamma*sigmaDot*abs(sigmaDot)-sigma;
            

            derivs = [ddelta; ddeltaDot; dsigma; dsigmaDot; dY; dYDot];
        end
end