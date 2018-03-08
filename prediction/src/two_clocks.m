function clocks
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
angle1 = 0.15;
velangle1 = -0.1;
angle2 = 0.2;
velangle2 = 0;
y = 0;
ydot = 0;
% Convert to the sum and difference
delta = angle1 - angle2;
deltadot = velangle1 - velangle2;
sigma = angle1 + angle2;
sigmadot = velangle1 + velangle2;
% pack them into an appropriate vector
init = [delta;deltadot;sigma;sigmadot;y;ydot];
% Define the start and end times
Tstart=0;
Tend=1000;
% Define the events function
options = odeset('Events',@escapement);
% Set the figure window
clf
while Tstart < Tend
    % Solve the ODEs until there is an event
    [t,vars,te,ye,ie]=ode45(@clockodes,[Tstart Tend],init,options);
    % Get the angles and velocities
    angle1 = (vars(:,1)+vars(:,3))/2;
    angle2 = (vars(:,3)-vars(:,1))/2;
    velangle1 = (vars(:,2)+vars(:,4))/2;
    velangle2 = (vars(:,4)-vars(:,2))/2;
    plot(t,angle1,'r'), hold on
    plot(t,angle2,'b')
    plot(t,vars(:,5),'k')
    % Update the initial conditions
    init = vars(end,:)';
    % Get the angle velocities
    anglevel1 = (init(2)+init(4))/2;
    anglevel2 = (init(4)-init(2))/2;
    % Determine how many events
    numevs = length(ie);
    % Engage the kick
%     for j = 1:numevs
%         state = ie(j);
%         if state==1
%             anglevel1 = -(1-c)*anglevel1 - e;
%         end
%         if state==3
%             anglevel1 = -(1-c)*anglevel1 + e;
%         end
%         if state==2
%             anglevel2 = -(1-c)*anglevel2 - e;
%         end
%         if state==4
%             anglevel2 = -(1-c)*anglevel2 + e;
%         end
%     end
%     % update the sum and difference
%     init(2) = anglevel1-anglevel2;
%     init(4) = anglevel1+anglevel2;
%     % Update the start time
    Tstart = t(end);
end
eig(A)

    function derivs = clockodes(t,vars)
        derivs = A*vars;
    end

    function[value,isterminal,direction]=escapement(t,vars)
        value = [(vars(1)+vars(3))/2-Phi; (vars(3)-vars(1))/2-Phi;
            (vars(1)+vars(3))/2+Phi; (vars(3)-vars(1))/2+Phi];
        isterminal = [1;1;1;1];
        direction = [1;1;-1;-1];
    end
end