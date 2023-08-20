function Problem = bratu(m,alpha,lambda)
%% Bratu's problem
%    PDE:
%        u_xx + u_yy + alpha * u_x + lambda * exp(u) = 0
%        with Dirichlet boundary conditions:
%        u(x,y) = 0 on unit square domain
%
%    Finite Difference Method:
%        D * u + alpha * J * u + lambda * exp(u) = 0
%
%    Input:
%        m = num. of pts on grid side, i.e. grid is m*m + boundary
%        alpha, lambda = coefficient scalar


    %%--- grid parameters
    h = 1.0/(m+1); %%--- x, y in [0,1]
    ah = alpha*h*0.5;
    lh = lambda*h*h;
    %%--- discretization operator
    A = fd3d(m, m, 1, ah, 0.0, 0.0, 0.0);
    %%--- problem struct
    Problem.name = @() 'Finite_Difference_Method_on_Bratu';
    Problem.grid = @() [num2str(m) 'x' num2str(m)];
    Problem.alpha = @() alpha;
    Problem.lambda = @() lambda;
    Problem.A = @() A;
    
    %%--- dim
    Problem.dim = @() m*m;
    Problem.samples = @() 0;

    %%--- cost
    Problem.cost = @cost;
    function c = cost(u)
        c = norm(grad(u));
    end

    %%--- gradient
    Problem.grad = @grad;
    function g = grad(u)
        g = A*u - lh*exp(u);
        %g = A'*g - lh*exp(u).*g;
    end

    %%--- hessian
    Problem.hess = @hess;
    function H = hess(u)
        H = A - lh*diag(exp(u));
        H = sparse(H);
    end
end

%% 3d Laplacian
function A = fd3d(nx,ny,nz,alpx,alpy,alpz,dshift)
    tx = tridiag(2, -1+alpx, -1-alpx, nx) ;
    ty = tridiag(2, -1+alpy, -1-alpy, ny) ;
    tz = tridiag(2, -1+alpz, -1-alpz, nz) ;
    A = kron(speye(ny,ny),tx);
    if(ny > 1)
        A = A + kron(ty,speye(nx,nx)); 
    end
    if (nz > 1) 
         A = kron(speye(nz,nz),A) + kron(tz,speye(nx*ny,nx*ny)); 
    end
    A = A - dshift * speye(nx*ny*nz,nx*ny*nz);
    
    function T = tridiag(a, b, c, n)
        sub = diag(ones(n-1,1),-1);
        T = a*eye(n) + b .* sub + c .* sub';
    end
end