function two_pendulums
    clear all
    close all
    
    % System parameters
    m = 0.0605   % mass of bob in kg
    M = 1.3280   % mass of cart in kg
    l = 0.327    % pendulm length in m
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
    omega = K/(M+2*m)*sqrt(g/l);       % Dimensionless poarameter for cart restoring force
    mu = m/(M + 2*m);                  % Cart to bob mass ratio
    
    % Define the matrix
    
    A1 = [1 0 0 0 0 0;
          0 1 0 0 0 1;
          0 0 1 0 0 0;
          0 0 0 1 0 1;
          0 0 0 0 1 0;
          0 mu 0 mu 0 1];
      
    A2 = [0 1 0 0 0 0;
          -1 -2*gamma 0 0 0 0;
          0 0 0 1 0 0;
          0 0 -1 -2*gamma 0 0;
          0 0 0 0 0 1;
          0 0 0 0 0 -2*Gamma];
      
    A = inv(A1) * A2;
    
    A = [0 1 0 0 0 0;
        -1 -2*gamma 0 0 0 0;
        0 0 0 1 0 0;
        0 0 -1/(1-2*mu) -2*gamma/(1-2*mu) 2*omega^2/(1-2*mu) 4*Gamma/(1-2*mu);
        0 0 0 0 0 1;
        0 0 mu/(1-2*mu) 2*mu*gamma/(1-2*mu) -omega^2/(1-2*mu) -2*Gamma/(1-2*mu)]
    
    inv(A1) * A2
   
    
    % Define the initial conditions (using original variables)
    theta1 = 0.15;
    thetaDot1 = 0;
    theta2 = -0.15;
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
    Tend=300;
    
    tspan = [Tstart, Tend];


    [T,M]=ode45(@clockodes,tspan,init);
    
    % Get the angles and velocities
    theta1 = (M(:,1)+M(:,3))/2;
    theta2 = (M(:,3)-M(:,1))/2;
    cart = M(:,5);
    thetaDot1 = (M(:,2)+M(:,4))/2;
    thetaDot2 = (M(:,4)-M(:,2))/2;

    plot(T, theta1, 'r'), hold on
    plot(T, theta2, 'b')
    plot(T, cart, 'k')
    
    legend('Pendulum 1','Pendulum 2','Cart')




        function derivs = clockodes(t,W)
            derivs = A*W;
        end
end