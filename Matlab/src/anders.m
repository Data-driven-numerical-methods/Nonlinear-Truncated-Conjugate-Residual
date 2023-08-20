   function [x,FVAL,history] = anders(x, Problem, param)
%% function [x,FVAL,history] = anders(x, FF, param)  
%% rewritten in style of solver - where  
%% FF(x) = gradient -- Fixed point mapp g(x) = x - mu*F(x)
%% [so in Anderson f(x) = g(x)-x = -mu*F(x) = -mu*gradient]
%% note: if we define f(x) = x-g(x) below then  beta must 
%% change sign (beta<0)
%% mu is passed in param along with beta, restart, itermax, 
%% nw, tol, and b_savlist (logical) 
%%-------------------- unpack  
  tol =  param.tol;
  itmax = param.max_iter; 
  nw = param.mem_size;
  mu = param.mu;
  beta = param.beta;
  b_savhist = param.savsols ;
%--- extract funcitons
if isfield(Problem, 'full_grad')
    FF = @(x) Problem.full_grad(x);
elseif isfield(Problem, 'grad')
    FF = @(x) Problem.grad(x);
end
if ~isfield(Problem, 'cost')
    cost = @(x) norm(FF(x));
else
    cost = @(x) Problem.cost(x);
end
%%-------------------- start  
  DX = [];
  DF = [];
  history = [];
  m = 0;
  FVAL = [];
%%--------------------initialize
  val = cost(x);
  d = FF(x);
  gx = x-mu*d;       %% x - mu * x = x - mu * grad 
  f = gx-x;          %% = -mu*d = -mu*grad
  oldx = x;
  oldf = f;
%%  beta damping. [note: change sign of beta if fx=x-gx]
  x = x + beta*f;
  rhog = norm(d) ;
%% -------------------- return if norm(F) == 0
  if (rhog == 0), return, end
  tol1 = tol*rhog;  
%%%
  FVAL = [FVAL, val];
  m = 0;
  for it = 1:itmax
    if (b_savhist) history = [history, x];, end    
%%-------------------- call FF (`gradient')     
    val = cost(x);
    d = FF(x);
    gx = x-mu*d;      
    f = gx-x;         
    FVAL = [FVAL, val];
%%-------------------- save differences
    m = m+1;
    DX(:,m) = x-oldx;
    DF(:,m) = f-oldf;
    oldx = x;
    oldf = f;
%%-------------------- loop
    rhog = norm(d);
    if (rhog == 0), return, end
    fprintf('it %3d  ff %e   rho %e\n',it, val,rhog);
%%-------------------- convergence test     
    if (rhog < tol1), return, end
%-------------------- use truncated svd or some other regul.
    gam = pinv(DF(:,1:m),1.e-13)*f;
    y   = x - DX(:,1:m)*gam;
%%-------------------- beta damping. [note: change sign of beta if fx=x-gx]
    x  = y + beta*(f - DF(:,1:m)*gam);
%%-------------------- move down all vectors when size exceeds nw
    if (m > nw) 
      DX = DX(:, 2:end);
      DF = DF(:, 2:end);
      m = nw; 
    end
  end
 
