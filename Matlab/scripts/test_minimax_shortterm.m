close all;
clear all;
clc;
n = 50;
A = randn(n);
A = A'*A+ 1*eye(n);
A = A/norm(A);

b = randn(n,1); 
c = randn(n,1);

xinit = randn(n,1);
yinit = randn(n,1);

fp0 = [xinit;yinit];
xtrue = -A'\c;
ytrue = -A\b;

stepsize = 1;
reg = 0.99;
problem = minimax(A,b,c,[xtrue;ytrue], stepsize, reg);
x_opt = [xtrue;ytrue];
F = @(x) problem.Agrad(x);
problem.sol_opt = x_opt;
%%-------------------- set some params
itmax =1000-1;

disp('testing nltgcr')
tol = 1.e-13; 
lb = 1; restart = itmax;
tic
problem = minimax(A,b,c,[xtrue;ytrue], stepsize, reg);
[sol1, fval1, cost1, ~, ~]= nltgcr2(F,fp0,lb,tol,itmax, problem, restart, 1,0);
toc
t1 = toc;

%%-------------------- nltgcr2 
disp('testing nltgcr')
lb = 5; restart = itmax;
tic
[~, fval2, ~, ~, ~]= nltgcr2(F,fp0,lb,tol,itmax, problem, restart,1,0);
toc
t2 = toc;




print = 100;
F_GDA = @(x) altGDA(x,n,1, A, b, c);
[~,~,res_aqr, rest_aqr] = walkerQR(F_GDA,[xtrue;ytrue], fp0,1,itmax,tol, print);

print = 100;
F_GDA = @(x) altGDA(x,n,1, A, b, c);
[~,~,res_aqr2, rest_aqr2] = walkerQR(F_GDA,[xtrue;ytrue], fp0,5,itmax,tol, print);

figure(1)
cyan        = [0.2 0.8 0.8];
brown       = [0.2 0 0];
orange      = [1 0.5 0];
blue        = [0 0.5 1];
green       = [0 0.6 0.3];
red         = [1 0.2 0.2];
purple = [0.4940, 0.1840, 0.5560];



interval = 100;
start=1;
semilogy(fval1(start:end,1),fval1(start:end,2), '-o', 'color',green,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(fval1))
hold on
semilogy(fval2(start:end,1),fval2(start:end,2), '-+','color',red,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(fval2))
semilogy(res_aqr(:,1),rest_aqr(:,2),'--o','color',green,'linewidth',1,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_aqr))
semilogy(res_aqr2(:,1),rest_aqr2(:,2),'--+','color',brown,'linewidth',1,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_aqr2))

h1=legend('TGCR [1,mv]','TGCR [5,mv]');
set(h1,'fontsize',12)
hold off
set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',12)
xlabel('Iteration')
ylabel('Log distance to optimal')
f = gcf;
exportgraphics(f,'results/minimax_alt_rand1.png','Resolution',300)