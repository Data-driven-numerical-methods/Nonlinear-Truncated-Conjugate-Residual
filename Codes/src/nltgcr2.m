function [sol, fval, cost, P,AP]= nltgcr2(FF,sol,lb,tol,itmax, problem, restart, epsf, v)
%% function [sol, res, cost, P,AP]= nltgcr2(FF,sol,lb,tol,itmax, problem, restart)  
%% NOTEs: lb defines number of vectors kept - [symmetric case lb == 2]
%%        restart defines restart dimension -- re restart every restart steps
%%        problem now contains sol_opt
%%-------------------- initialize
%  sol_opt = problem.sol_opt;

mom = 0;
n = length(sol); 
%%-------------------- define P and AP to contain zero columns
P  = zeros(n,lb);
AP = zeros(n,lb);
%%--------------------get initial residual vector and norm 
r = FF(sol);         %% y - A*sol;
rho = norm(r);
tol1 = tol*rho; 
%%-------------------- Ar + normalize    FF = b-Ax --> 
% ep = epsf*norm(sol)/rho;
%  Ar = (FF(sol-ep*r) - r)/ep;
ep = 1e-15;
imagi= sqrt(-1);
Ar = imag(FF(sol-ep*r*imagi)/ep);
t = norm(Ar);
t = 1.0 / t;
P(:,1) = t * r;
AP(:,1) = t*Ar;
it = 0;
fval = [1,problem.cost(sol)];
cost = 0;
cost(1) = rho;
rr0 = rho;
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

    alph = AP'*r;
    
    mom = -P * alph + v * mom;
    sol = sol - mom;
    r  = FF(sol);
    
    %    r = r -alph*AP(:,i2);
    %%## NOTE: ALTERNATIVE    r defined as r := r -alph*AP(:,j) --> one less feval
    %% but not stable + not good theoretical support for this. 
    rho = norm(r);
    if mod(it, 1) ==0
       cost = problem.cost(sol);
       fval = [fval;[it,cost]];
       fprintf(1,' it %d  distance to optimal %10.3e \n', it, cost);
    end
    Ar = imag(FF(sol-ep*r*imagi)/ep);
    %% || Ar || / ep  ~ || FF(u+ep*r)- FF(u) || 
    %%--------------------orthonormnalize  Ap's
    p  = r;
    if (i <= lb), k = 0; else, k=i2; end
    while(1) 
        %%---------- define next column - circular storage to avoid copying
        if (k  ==  lb), k=0; end
        k = k+1;
        tau = dot(Ar,AP(:,k));
        %     disp(tau)
        p = p-tau*P(:,k);
%         Ar = imag(FF(sol-ep*r*imagi)/ep);
        Ar = Ar-tau* AP(:,k);
        %%---------- update u (last column of current Hess. matrix)
        if (k == i2), break; end
    end
    t = norm(Ar);
    %%-------------------- Now  Ar==Ap. If   Ap == 0 can't advance         
    if (t == 0.0)
        return; 
    end
    %%-------------------- we restart every `restart' iterations
    if (mod(it,restart) == 0) 
        i2 = 0;
        i  = 0;
        P  = zeros(n,lb);
        AP = zeros(n,lb);
        %%--------------------initial residual vector and norm 
        r  = FF(sol);
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
    AP(:,i2) = t*Ar;
    P(:,i2) = p*t;
end
