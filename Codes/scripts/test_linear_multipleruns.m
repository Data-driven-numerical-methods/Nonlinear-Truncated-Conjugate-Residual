clear all; close all; clc
% Load Dataset
TGCR = [];
GMRES = [];
AA = [];
for e = 1:50     
    n = 100;
    A = randn(n, n);
    A = A'*A -0.01*eye(n);
    b = randn(n,1);
    stepsize=1;
    n = size(A,1);
    x00 = randn(n,1);
    x0 = x00;
    
    x_opt = A\b;
    problem = linearsys(A, b, stepsize);
    % Define FF
    F = @(x) problem.grad(x);
    problem.sol_opt = x_opt;
    %%-------------------- set some params
    itmax = n-1;
    x0  = x00;
    k= 0;
    %%-------------------- nltgcr2 
    disp('testing nltgcr')
    tol = 1.e-09; 
    lb = 1; restart = 100;
    epsf = 1;
    tic
    [sol1, fval1, ~, ~, ~]= nltgcr2(F,x00,lb,tol,itmax, problem, restart, epsf,0);
    toc
    t1 = toc;
    TGCR = [TGCR, fval1 ];

    
    tic
    [x,fl0,rr0,it0,rv0] = gmres(A,b,[],1e-20,n,[],[],x00);
    toc
    t4 = toc;
    
    GMRES = [GMRES, rv0];
    
    tic
    data.DX = [];  data.DF = []; data.nv = 5; data.beta = 1.0; 
    xa = x00;
    restart = data.nv;
    k=1;
    tol = 1.e-08;
    cost4(1) = norm(xa-x_opt);
    fval4(1) = problem.cost(xa);
    disp('Anderson sith svd')
    stepsize = 0.001;
    for it = 0 : itmax-1
        xa1 = xa - stepsize * problem.grad(xa);
        k=k+1;
        rho = norm(xa1-x_opt);
        cost4(k)  = rho;
        fval4(k) = problem.cost(xa1);
        fprintf(1,' it %3d distance to optimal %10.3e\n',it, rho)
        if (it == 0), tol1 = tol* rho; end 
        if (rho < tol1 || isnan(rho) || rho==Inf)
            disp(' stop') 
            break
        end
        [xa, data] = anders1(xa,xa1,data);
        if (mod(it, restart)== 0)
            data.DX = []; data.DF = [];
        end
    end
    toc
    taa =toc;
    AA = [AA, fval4'];
end

figure(1)
cyan        = [0.2 0.8 0.8];
brown       = [0.2 0 0];
orange      = [1 0.5 0];
blue        = [0 0.5 1];
green       = [0 0.6 0.3];
red         = [1 0.2 0.2];
purple      = [0.4940, 0.1840, 0.5560];
interval = 10;
%% It depicts distance to the correct solution as a function of iteration. 


figure(1)
start = 1;
for e = 1:50
    h(1) = semilogy(TGCR(start:end,2*e-1),TGCR(start:end,2*e), '-o', 'color',green,'linewidth',1,'MarkerSize',3,'MarkerIndices', 1:interval:length(fval1));
    hold on
   
    h(2) = semilogy(AA(start:end,e), '-s','color',cyan,'linewidth',1,'MarkerSize',3,'MarkerIndices', 1:interval:length(fval4));
    
    h(3) = semilogy(GMRES(1:end-1, e),'--','color',orange,'linewidth',1,'MarkerSize',3,'MarkerIndices', 1:interval:length(rv0)-1);
end

legend(h,{'TGCR [1,mv]','AA m=10','Full GMRES'},'Location', 'Best')
hold off
set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',12)
xlabel('Iteration')
ylabel('Log Residual Norm')
f = gcf;
exportgraphics(f,'linear_all2.png','Resolution',300)