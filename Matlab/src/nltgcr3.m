  function [sol, fval, acc, P,AP]= nltgcr3(FF,sol,lb,tol,itmax, problem, restart, epsf, v)
%% function [sol, res, cost, P,AP]= nltgcr2(FF,sol,lb,tol,itmax, problem, restart)  
%% NOTEs: lb defines number of vectors kept - [symmetric case lb == 2]
%%        restart defines restart dimension -- re restart every restart steps
%%        problem now contains sol_opt
 mom = 0;
 n = length(sol); 
%%-------------------- define P and AP to contain zero columns
 P  = zeros(n,lb);
 AP = zeros(n,lb);
%%--------------------get initial residual vector and norm 
 r = FF(sol);        
 rho = norm(r);
 tol1 = tol*rho; 
 %%-------------------- Ar + normalize    FF = b-Ax --> 
ep = 1e-13;
imagi= sqrt(-1);
Ar = imag(FF(sol-ep*r*imagi)/ep);
 t = norm(Ar);
 t = 1.0 / t;
 P(:,1) = r*t;
 AP(:,1) = Ar*t;    
 it = 0;
 fval = [1,problem.cost(sol)];
 pred = problem.prediction(sol);
 acc = [1,problem.accuracy(pred)];
 
%%
 fprintf(1,' it %d  rho %10.3e \n', it,rho);
 if (rho <= 0.0)
   return
 end
%%--------------------get abs residual norm tol from 1st residual norm
%% --main loop: i = loop index. it # steps
%% i2 points to current column in P, AP. Cycling storage used 
 i2= 1;
 i = 1;
 for it =1:itmax 
    if epsf ==1
        alph = AP'*r;
        %% if v > 0, it's momentum
        mom = -P * alph + v * mom;
        sol = sol - mom;
        r  = FF(sol);
    else

      % This is the original version: 
        alph = dot(r, AP(:,i2));
        sol = sol + alph * P(:,i2);
        % r  = FF(sol);
        r = r -alph * AP(:,i2);
    end
%    r = r -alph*AP(:,i2);
   %%## NOTE: ALTERNATIVE    r defined as r := r -alph*AP(:,j) --> one less feval
   %% but not stable + not good theoretical support for this. 
   rho = norm(r);
   if mod(it, 10) ==0
       cost = problem.cost(sol);
       fval = [fval;[it,cost]];
       pred = problem.prediction(sol);
       test_acc = problem.accuracy(pred);
       acc = [acc;[it,test_acc]];
       fprintf(1,' it %d  function val %10.3e, acc is %10.3e \n', it, cost, test_acc);
   end
   Ar = imag(FF(sol-ep*r*imagi)/ep);
  

   p  = r;
   if (i <= lb), k = 0;, else, k=i2;, end
   while(1) 
   %%---------- define next column - circular storage to avoid copying
    if (k  ==  lb), k=0;, end
    k = k+1;
    tau = dot(Ar,AP(:,k));
    p = p-tau*P(:,k);
    Ar = Ar-tau* AP(:,k);
    %%---------- update u (last column of current Hess. matrix)
    if (k == i2), break; end
   end
  t = norm(Ar);
%%-------------------- Now  Ar==Ap. If   Ap == 0 can't advance         
  if (t == 0.0), return;  end
  %%-------------------- we restart every `restart' iterations
  if (mod(it,restart) == 0) 
    i2 = 0;
    i  = 0;
    P  = zeros(n,lb);
    AP = zeros(n,lb);
%%--------------------initial residual vector and norm 
    r  = FF(sol);
    rho = norm(r);
    mom = 0;
    Ar = imag(FF(sol-ep*r*imagi)/ep);
    t  = norm(Ar);
    p  = r;

  end
%%-------------------- truncate subspace  
  if (i2  == lb), i2=0; end
  i2=i2+1;
  i = i+1;
  t = 1.0 / t;
  AP(:,i2) = Ar*t;
  P(:,i2) = p*t;
 end
