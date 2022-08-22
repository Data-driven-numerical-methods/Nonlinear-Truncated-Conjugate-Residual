function [z,gz] = myfun1(t)
x = t(1);
y = t(2);
z = (x-3)^2+y^2;
if nargout > 1 % If a gradient is called for
    gz = [2*(x-3);2*y];
end

