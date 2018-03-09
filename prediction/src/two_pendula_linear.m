function clocks
    clear all
    close all
    
    % Set the clock parameters
    Phi = 0.1;
    gamma = 0.01;
    c = 1;
    e = 0.05;
    
    % Set the platform parameters
    Gamma = 0.1;
    omega = 0.5;
    mu = 0.02;
    
    % Define the matrix
    A = [0 1 0 0 0 0;
        -1 -2*gamma 0 0 0 0;
        0 0 0 1 0 0;
        0 0 -1/(1-2*mu) -2*gamma/(1-2*mu) 2*omega^2/(1-2*mu) 4*Gamma/(1-2*mu);
        0 0 0 0 0 1;
        0 0 mu/(1-2*mu) 2*mu*gamma/(1-2*mu) -omega^2/(1-2*mu) -2*Gamma/(1-2*mu)];
    
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