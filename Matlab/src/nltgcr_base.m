function [sol,varargout] = nltgcr_base(sol, Problem, params)
%% Nonlinear Truncated GCR
%
% function [sol, varargout] = nltgcr_base(sol,Problem,params)
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
%           safeguard= threshold for auto-restart (default: 1.e+3)
%
% output:
%        sol       = solution
%        FVAL      = tracking of function values
%        FUNC_EVAL = accumulated number of function evaluations
%
% reference:    https://arxiv.org/abs/2306.00325

  format long

%% --- extract parameters
if ~isfield(params, 'mem_size')
    lb = 10;
else
    lb = params.mem_size;
end

if ~isfield(params, 'tol')
    tol = 1.0e-12;
else
    tol = params.tol; 
end

if ~isfield(params, 'max_iter') %--- max number of iterations
    itmax = 100;
else
    itmax = params.max_iter;
end

if ~isfield(params, 'max_ls') %--- max number of line search steps
    nbtmax = 10;
else
    nbtmax = params.max_ls;
end

if ~isfield(params, 'restart')
    restart = lb*5;
else
    restart = params.restart;
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

if ~isfield(params, 'safeguard')
    safeguard = 1.e+3;
else
    safeguard = params.safeguard;
end

%%-------------------- extract functions
FF = @(x) Problem.grad(x);
fun = @(x) Problem.cost(x);

nfe = 0;       %---- num. of function evaluation
FVAL = [fun(sol)];  %---- tracking function values
FUNC_EVAL = [0];        %---- tracking num. of func eval
%%-------------------- set nbtmax = max number of backtracking steps
lam0 = stepsize;
%%-------------------- tolx = smallest possible variation to sol
n = length(sol); 
%%-------------------- define P and AP to contain zero columns
P  = zeros(n,lb);
AP = zeros(n,lb);
xrec = zeros(lb,1);
%%--------------------get initial residual vector and norm 
r = -FF(sol);   nfe = nfe + 1;
rho = norm(r);
tol1 = tol*rho; 
ep = 1d-10;
imagi= 1i;
p = r;
Ar = imag(FF(sol+ep*p*imagi)/ep);   nfe = nfe + 1;
t = norm(Ar);
t = 1.0 / t;
P(:,1) = t * r;
AP(:,1) = t*Ar;
xrec(1) = t * norm(p,'inf');
it = 0;
prevf = FVAL;				%
fprintf(1,' ** it %d  rho %10.3e \n', it,rho);
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
	 gg =-gg;
     tc =-tc;
    end
%%-------------------- backtracking loop 
    nbt = 0;
    lam = lam0;

    while (nbt<nbtmax)
      nbt  = nbt+1;
      sol_tmp = sol +lam*gg;
      fun_val = fun(sol_tmp);
      r = -FF(sol_tmp);   nfe = nfe + 1;
      if (fun_val < prevf - ca*tc*lam), break; end
      lam = lam*.5;
    end
%%-------------------- adjust init stepsize for line search
    if nbt >= 2
        lam0 = 0.5*lam;
    else
        lam0 = min(1.0,2*lam);
    end
%%-------------------- print info 
    fprintf(1,' rT * gg = %7.3e   back steps= %d\n',tc,nbt)
    rho = norm(r);
    if (rho <= tol1)
        disp('grad converged')
        break
    end
    sol = sol_tmp;
    prevf = fun_val;
    FVAL(it+1) = fun_val;
    FUNC_EVAL(it+1) = nfe;
%%--------------------
    fprintf('   %5d    %10.3e    %10.3e\n', it, fun_val, rho);
%%--------------------orthonormnalize  Ap's
    p  = r;
    w  = norm(p,'inf');
    Ar = imag(FF(sol+ep*p*imagi)/ep);  nfe = nfe + 1;
    vv = Ar;
    if (i <= lb), k = 0; else, k=i2; end
    while(1) 
%%---------- define next column - circular storage to avoid copying
        if (k  ==  lb), k=0; end
        k = k+1;
        tau = dot(Ar,AP(:,k));
        p = p-tau*P(:,k);
	    Ar = Ar - tau*AP(:,k);
        w = w + abs(tau)*xrec(k);
%%---------- update u (last column of current Hess. matrix)
        if (k == i2), break; end
    end
    t = norm(Ar);
    w = w / t;

%%-------------------- Now  Ar==Ap. If   Ap == 0 can't advance         
    if (t < rho*1.e-16)
      disp(' *** Ar = 0 ----> restart')
      break
    end

    if abs(FVAL(end) - FVAL(end-1)) < 1.e-16*abs(FVAL(1))
        disp('lack of progress -----> stop')
        break
    end
%%-------------------- we restart every `restart' iterations
    if (mod(it,restart) == 0 || w > safeguard)  
      fprintf('restart\n')
      i2 = 0;
      i  = 0;
      P  = zeros(n,lb);
      AP = zeros(n,lb);
      xrec = zeros(lb,1);
%%--------------------initial residual vector and norm
      p = r;
      Ar = vv;
      t  = norm(Ar);
      w = norm(p,'inf') / t;
    end
%%-------------------- truncate subspace  
    if (i2  == lb), i2=0; end
    i2=i2+1;
    i = i+1;
    t = 1.0 / t;
    AP(:,i2) = t*Ar;
    P(:,i2) = p*t;
    xrec(i2) = w;
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
