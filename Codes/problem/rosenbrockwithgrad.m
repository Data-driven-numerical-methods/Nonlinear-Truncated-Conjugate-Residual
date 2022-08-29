function [f,g,H] = rosenbrockwithgrad(x)
% Calculate objective f
f = 100*(x(2) - x(1)^2)^2 + (1-x(1))^2;

if nargout > 1 % gradient required
    g = [-400*(x(2)-x(1)^2)*x(1) - 2*(1-x(1));
        200*(x(2)-x(1)^2)];
    H = [-400*x(2) + 1200*x(1)^2 + 2, -400*x(1);
        -400*x(1), 200];
end
end

