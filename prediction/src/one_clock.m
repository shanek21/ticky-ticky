function one_clock()

x0 = [0.75;0];
l = 0.34;
g = 9.8;
period = sqrt(l/g);

[T,X] = ode45(@derivs,[0 10000], x0);
plot(T*period,X(:,1), 'b')
xlim([0 350])
%figure
%plot(X(:,1),X(:,2), 'r')

end

function dxdt = derivs(t,x)
    theta = x(1);
    omega = x(2);
    g = 0.000858;
    Fescape = 0;
    
   % if -0.75<theta && theta<0.75 && sign(omega)== 1
    %    Fescape = 0.01;
   % end
    
    dxdt = [omega; (-4*sin(theta)-2*g*(omega)*abs(omega)+ Fescape)];

end