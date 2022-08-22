function [fp] = altGDA(x,n,lr,A,b,c)
 x0 = x(1:n);
 y0 = x(n + 1 : end);
 gx = A * y0 + b ; 
 x1 = x0 - lr * gx ;
 gy = A'* x1 + c;
 y1 = y0 + lr * gy ;  
 fp = [x1;y1];
end