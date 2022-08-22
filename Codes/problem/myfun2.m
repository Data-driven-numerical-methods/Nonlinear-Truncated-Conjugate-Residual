function [z,gz] = myfun2(x)

z = ( x(1) - 2 )^4+ ( ( x(1) - 2 ) * x(2) )^2 + ( x(2) + 1 )^2;

if nargout > 1 % If a gradient is called for
      g1 = 4 * ( x(1) - 2 )^3 ...
           + 2 * ( x(1) - 2 ) * x(2)^2;

      g2 = 2 * ( x(1) - 2 )^2 * x(2)...
           + 2 * ( x(2) + 1 );
      gz = [g1;g2];
end
