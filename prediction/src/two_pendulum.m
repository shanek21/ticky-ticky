function two_pendulum()
% clear all
% close all
hold on
hold all
phi1_0 = pi/2;
phi2_0 = pi/2;
omega1_0 = 0;
omega2_0 = 0;
Y_0 = 1;
V_0 = 0;
l = 0.32;
g = 9.8;
m = 121;
M = 1328;
mu = m/(M + 2*m);
b = .0008;
gamma = b*sqrt(l/(4*g));
B = .0065;
Gamma = B*sqrt(l/(4*g))*(M+2*m);
period = sqrt(l/g);

x0 = [phi1_0; omega1_0; phi2_0; omega2_0; Y_0; V_0];  %%inition condition

[T, X] = ode45(@derivs, [0 10000], x0);


figure(1) %angle 1 over time
plot(T*period,X(:,1), 'b')
plot(T*period, X(:,3), 'r')
xlim([0 300])
xlabel('Time (seconds)')
ylabel('Phi (rad)')
title('Position of Pendulum')
legend('Phi 1', 'Phi 2')

figure(2) %angle 1 over time
plot(T*period,X(:,2), 'b')
plot(T*period, X(:,4), 'r')
xlim([0 300])
xlabel('Time (seconds)')
ylabel('omega (rad/s)')
title('Velocity of Pendulum')

% figure(3) %Position vs Acceleration
% plot(X(:,1), X(:,2), 'r')
% xlabel('Position (theta)')
% ylabel('Acceleration (rad/s)')
% title('Position vs Acceleration')

%% calculate the right hand side of the ode solver
function res = derivs(t, x)
    phi1 = x(1);
    omega1 = x(2);
    phi2 = x(3);
    omega2 = x(4);
    Y = x(5);
    V = x(6);
    
    dphi1dt = omega1;
    dphi2dt = omega2;
    domega1dt = (2*Gamma*V - mu*(sin(phi1) + sin(phi2)))*cos(phi1) + 2*gamma*omega1*abs(omega1) + sin(phi1);
    domega2dt = (2*Gamma*V - mu*(sin(phi1) + sin(phi2)))*cos(phi2) + 2*gamma*omega2*abs(omega2) + sin(phi2);
    dYdt = V;
    dVdt = -2*Gamma*V + mu*(sin(phi1) + sin(phi2));
    
    res = [dphi1dt; domega1dt; dphi2dt; domega2dt; dYdt; dVdt];
end
end