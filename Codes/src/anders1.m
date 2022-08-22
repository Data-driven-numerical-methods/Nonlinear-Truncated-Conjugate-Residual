   function [y, data] = anders1(x, gx, data)
%% function [y] = anders(x, gx, it, nv, beta)
%% anderson acceleration - in its simplest form.
%% uses truncated SVD for LS problem.      
%% This code takes any pair x, g(x) and performs an 
%% Anderson acceleration --
 nv = data.nv;
 DX = data.DX;
 DF = data.DF;
 beta = data.beta;
%%-------------------- new x,f
 f = gx-x;
 [n, m] = size(DX);
%%-------------------- save f for next call
 m1 = m+1;
%%-------------------- save x, f for next call
 DX(:,m1) = x;
 DF(:,m1) = f;
%%-------------------- very first time just return gx
 if (m == 0)
   data.DX = DX;
   data.DF = DF;
   y = x + beta*f;
   return;
 end
%%-------------------- update DX and DF -- most recent x, f were stored
%%                     in m-th columns of DX and DF 
 DX(:,m) = x - DX(:,m) ;
 DF(:,m) = f - DF(:,m) ;
%-------------------- use truncated svd or some other regul.
%gam = pinv(DF(:,1:m),1.e-12)*f;
 gam = pinv(DF(:,1:m),1.e-13)*f;
%%gam = pinv(DF(:,1:m))*f;
 y   = x - DX(:,1:m)*gam;
%%-------------------- case of a beta damping.
 y  = y + beta*(f - DF(:,1:m)*gam);
%%-------------------- move down all vectors when size exceeds nv
 if (m >= nv) 
   DX = DX(:, 2:end);
   DF = DF(:, 2:end);
 end
 data.DX = DX;
 data.DF = DF;
 %%-------------------- 
 
