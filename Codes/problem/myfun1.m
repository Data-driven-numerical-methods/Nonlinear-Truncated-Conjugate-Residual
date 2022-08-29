function [z,gz,H] = myfun1(t)
x = t(1);
y = t(2);
z = (y - cos(2*x) - ((x.^2)/10)).^2 + exp((x.^2 + y.^2)/100);
if nargout > 1 % If a gradient is called for
    gz1 = 2*(y - cos(2*x) - ((x.^2)/10))*(2*sin(2*x) - x/5)...
      + x/50*exp((x.^2 + y.^2)/100);
    gz2 = 2*(y - cos(2*x) - ((x.^2)/10)) + y/50*exp((x.^2 + y.^2)/100);
    gz = [gz1;gz2];
    h11 = (4*sin(2*x) - (2/5)*x)*(2*sin(2*x) - x/5) + 2*(y - cos(2*x) - ((x.^2)/10))*(4*cos(2*x) - 1/5)...
        + ((x/50)^2 + 1/50)*exp((x.^2 + y.^2)/100);
    h12 = 2*(2*sin(2*x) - x/5) +  (y/50)*(x/50)*exp((x.^2 + y.^2)/100);
    h22 = 2 + ((y/50)^2 + 1/50)*exp((x.^2 + y.^2)/100);
    H = [h11, h12;h12, h22];
end

