function [z,gz] = myfun3(t)
x = t(1);
y = t(2);
z = (x^2+y^2-2)^2;
if nargout > 1 % If a gradient is called for
    gz = [4*x*(x^2+y^2-1) ; 4*y*(x^2+y^2-1) ];
end
