function [sol,varargout] = nltgcr_linear(sol,Problem,params)
%% Nonlinear Truncated GCR w/ linear update
%
% function [sol, varargout] = nltgcr_linear(sol,Problem,params)
%
% inputs:
%        sol     = initial iterate
%        Problem = functions Problem.grad() and Problem.cost()
%        params  = parameters
%           mem_size = table size for saved vectors (default: 10)
%           tol      = tolerance of relative residual (default: 1e-12)
%           max_iter = max number of total iterations (default: 100)
%           max_ls   = max number of line search steps (default: 10)
%           restart  = max number of iterations for one restart period
%                      (default: 5*mem_size)
%           c        = control parameter for Armijo-Goldstein condition
%                      (default: 0.001)
%           stepsize = initial stepsize for line search (default: 1.0)
%
% output:
%        sol       = solution
%        FVAL      = tracking of function values
%        FUNC_EVAL = accumulated number of function evaluations
%
% reference:    https://arxiv.org/abs/2306.00325

  format long

%%-------------------- extract parameters
if ~isfield(params, 'mem_size')
    lb = 10;
else
    lb = params.mem_size;
end

if ~isfield(params, 'tol')
    tol = 1.e-12;
else
    tol = params.tol; 
end

if ~isfield(params, 'max_iter') %--- max number of iterations
    itmax = 100;
else
    itmax = params.max_iter;
end

if ~isfield(params, 'restart')
    restart = lb*5;
else
    restart = params.restart;
end

if ~isfield(params, 'max_ls') %--- max number of line search steps
    nbtmax = 10;
else
    nbtmax = params.max_ls;
end

if ~isfield(params, 'c') %--- control parameter for Armijo-Goldstein condition
    ca = 0.001;
else
    ca = params.c;
end

if ~isfield(params, 'stepsize') %--- initial stepsize in line search
    stepsize = 1.0;
else
    stepsize = params.stepsize;
end

%%-------------------- extract functions
FF = @(x) Problem.grad(x);
fun = @(x) Problem.cost(x);
%%-------------------- set eta = desired 'sufficient decrease'
nfe = 0;      %--- num. of function evaluation
FVAL = [fun(sol)];
FUNC_EVAL = [0];
%%-------------------- set nbtmax = max number of backtracking steps
lam0 = stepsize;
%%-------------------- tolx = smallest possible variation to sol
n = length(sol); 
%%-------------------- define P and AP to contain zero columns
P  = zeros(n,lb);
AP = zeros(n,lb);
%%--------------------get initial residual vector and norm 
r = -FF(sol);  nfe = nfe + 1;
rho = norm(r);
tol1 = tol*rho; 
ep = 1d-10;
imagi= 1i;
Ar = imag(FF(sol+ep*r*imagi)/ep);  nfe = nfe + 1;
t = norm(Ar);
t = 1.0 / t;
P(:,1) = t * r;
AP(:,1) = t*Ar;
it = 0;
fprintf(1,' it %d  rho %10.3e \n', it,rho);
%%--get abs residual norm tol from 1st residual norm
%%--main loop: i = loop index. it # steps
%%--i2 points to current column in P, AP. Cycling storage used 
i2= 1;
i = 1;
fprintf('   Iteration    fval       grad_norm\n');
for it =1:itmax 
    alph = AP'*r;
    gg = P * alph;
    tc = r'*gg; 
%%-------------------- change search direction in rare case of *not descent* 
    if (tc<0)
      fprintf('not descending --->> change dir\n')
      gg=-gg;
      tc=-tc;
    end
%%-------------------- backtracking loop 
    nbt = 0;
    lam  = lam0;
    while (nbt<nbtmax)
      nbt  = nbt+1;
      r_tmp = r - lam*(AP * alph);
      if (norm(r_tmp) < norm(r) - ca*tc*lam), break; end
      lam = lam*.5;
    end

    if nbt >= 2
        lam0 = 0.8*lam0;
    else
        lam0 = min(stepsize, 1.25*lam0);
    end
%%-------------------- print info 
    fprintf(1,' rT * gg = %7.3e   back steps= %d\n',tc,nbt)
    sol_tmp = sol +lam*gg;
    fun_val = fun(sol_tmp); %--- only used for recording, not computation
    r = r_tmp;
    rho = norm(r);  
    fprintf('   %5d    %10.3e    %10.3e\n', it, fun_val, rho);
    if (rho <= tol1)
        disp('grad converged')
        break
    end
%%-------------------- end info    
    sol = sol_tmp;
    FVAL(it+1) = fun_val;
    FUNC_EVAL(it+1) = nfe;
    
%%--------------------orthonormnalize  Ap's
    p  = r;
    Ar = imag(FF(sol+ep*p*imagi)/ep);  nfe = nfe + 1;
    if (i <= lb), k = 0; else, k=i2; end
    while(1) 
%%---------- define next column - circular storage to avoid copying
        if (k  ==  lb), k=0; end
        k = k+1;
        tau = dot(Ar,AP(:,k));
        p = p-tau*P(:,k);
        Ar = Ar - tau*AP(:,k);
%%---------- update u (last column of current Hess. matrix)
        if (k == i2), break; end
    end
    t = norm(Ar);
%%-------------------- Now  Ar==Ap. If   Ap == 0 can't advance     

    if abs(FVAL(end) - FVAL(end-1)) < 1.e-16*abs(FVAL(1))
        disp('lack of progress -----> stop')
        break
    end

%%-------------------- we restart every `restart' iterations
    if (mod(it,restart) == 0 || t < 1.e-16*rho) 
        restart_cond = true;
    else
        restart_cond = false;
    end
    if restart_cond 
        i2 = 0;
        i  = 0;
        P  = zeros(n,lb);
        AP = zeros(n,lb);
%%--------------------initial residual vector and norm 
        r = -FF(sol);  nfe = nfe + 1;
        p = r;
        Ar = imag(FF(sol+ep*p*imagi)/ep); nfe = nfe + 1;
        t  = norm(Ar);
    end
%%-------------------- truncate subspace  
    if (i2  == lb), i2=0; end
    i2=i2+1;
    i = i+1;
    t = 1.0 / t;
    AP(:,i2) = t*Ar;
    P(:,i2) = p*t;
end

nout = max(nargout,1)-1;
if nout == 1
    varargout(1) = {FVAL(:)};
end
if nout == 2
    varargout(1) = {FVAL(:)};
    varargout(2) = {FUNC_EVAL(:)};
end
end
