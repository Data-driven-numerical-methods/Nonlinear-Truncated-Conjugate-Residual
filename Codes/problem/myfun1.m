function [z,gz] = myfun1(t)
x = t(1);
y = t(2);
z = (y - cos(2*x) - ((x.^2)/10)).^2 + exp((x.^2 + y.^2)/100);
if nargout > 1 % If a gradient is called for
    gz1 = 2*(y - cos(2*x) - ((x.^2)/10)).^2*(2*sin(2*x) - x/5)...
      + x/50*exp((x.^2 + y.^2)/100);
    gz2 = 2*(y - cos(2*x) - ((x.^2)/10)) + y/50*exp((x.^2 + y.^2)/100);
    gz = [gz1;gz2];
end

