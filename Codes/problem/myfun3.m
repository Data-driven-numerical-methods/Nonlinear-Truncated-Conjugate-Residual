function [z,gz,H] = myfun3(t)
x = t(1);
y = t(2);
z = (x^2+y^2-2)^2;
if nargout > 1 % If a gradient is called for
    gz = [4*x*(x^2+y^2-2) ; 4*y*(x^2+y^2-2) ];
    H = [4*x*(x^2+y^2-2)+8*x^2, 8*x*y;
        8*x*y, 4*x*(x^2+y^2-2)+8*y^2];
end
