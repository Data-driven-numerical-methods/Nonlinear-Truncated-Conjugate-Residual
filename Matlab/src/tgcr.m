   function [sol, res, P,AP]= tgcr(A,sol,y,lb,tol,itmax)  
%% function [sol,P,AP]= tgcr(A,sol,y,lb,tol,itmax)
%% see dqgmres for meaning of parameters. 
%% truncated GCR/Orthomin -- Note lb does not quite have same meaning
%% as in dqgmres (we use lb 'p' vectors).
%%----------------------------------------------------------------------- 
%%-------------------- initialize 
   n = size(A,1);
   P  = zeros(n,lb);
   AP = zeros(n,lb);
%%--------------------get initial residual vector and norm 
   r  = y - A*sol;
   rho = norm(r);
   tol1 = tol*rho; 
%%-------------------- Ar + normalize
   Ar = A*r;
   t = norm(Ar);
   t = 1.0 / t;
   P(:,1) = r*t;
   AP(:,1) = Ar*t;    
   it = 0;
   res(1) = rho;
%%
   fprintf(1,' it %d  rho %10.3e \n', it,rho);
   if (rho <= 0.0)
     return
   end
%%--------------------get abs residual norm tol from 1st residual norm
%% --main loop: i = loop index. it # steps
%% i2 points to current column in P, AP. Cycling storage used 
   i2=1;
   for it =1:itmax 
     alph = dot(r,AP(:,i2));
     sol = sol+alph*P(:,i2);
     r   = r-alph*AP(:,i2);
     rho = norm(r);
     res(it+1) = rho;
     if (rho < tol1), break, end
     fprintf(1,' it %d  rho %10.3e \n', it,rho);
     Ar  = A*r;
%%--------------------orthonormnalize  Ap's
     p  = r;
     if (i <= lb), k = 0;, else, k=i2;, end
     while(1) 
%% ---------- define next column - circular storage to avoid copying
       if (k  ==  lb), k=0;, end
       k = k+1;
       tau = dot(Ar,AP(:,k));
       p = p-tau*P(:,k);
       Ar = Ar-tau* AP(:,k);
%%---------- update u (last column of current Hess. matrix)
       if (k == i2), break;, end
     end
     t = norm(Ar);
%%-------------------- Now  Ar==Ap. If   Ap == 0 can't advance       
     if (t == 0.0), return; , end
     if (i2  == lb), i2=0;, end
     i2=i2+1;
     AP(:,i2) = Ar/t;
     P(:,i2) = p/t;
   end
 
