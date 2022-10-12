close all;
clear all;
clc;

d = load('mnist.mat');
x_train = d.training.images;
y_train = d.training.labels;
x_test = d.test.images;
y_test = d.test.labels;
x_train=reshape(x_train(:,:,:),[],60000);
x_test = reshape(x_test(:,:,:),[],10000);
y_train = (y_train==1:10)';
y_test = (y_test==1:10)';
d = size(x_train,1);
n = length(y_train);
lambda = 0;
l = 10;

% Define Problem
problem = softmax_regression(x_train, y_train, x_test, y_test, lambda, l);
		    

x00 = randn(d*l,1);

stepsize = 1;
% Define FF
F = @(x) stepsize * problem.sgd_grad(x, 500);
%%-------------------- set some params
itmax = 500;
tol = 1.e-09;
x0  = x00;
k= 0;
%%-------------------- nltgcr2 
disp('testing nltgcr')
tol = 1.e-06; 
lb = 2; restart = 4;
tic
epsf = 1e-2;
[sol1, fval1, acc1, ~, ~]= nltgcr3(F,x00,lb,tol,itmax, problem, restart, epsf,0);
toc
t1 = toc;

%%-------------------- nltgcr2 
disp('testing nltgcr')
lb = 3; restart = 10;
tic
[sol2, fval2, acc2, ~, ~]= nltgcr3(F,x00,lb,tol,itmax, problem, restart, epsf,0.0);
toc
t2 = toc;

disp('testing f nltgcr')
F = @(x) stepsize * problem.full_grad(x);
lb = 1; restart = 1;
tic
[sol2, fval_gd, acc2, ~, ~]= nltgcr3(F,x00,lb,tol,itmax, problem, restart, epsf,0.0);
toc
t2 = toc;

%%-------------------- 
%% cost3: Anderson With SVD
data.DX = [];  data.DF = []; data.nv = lb; data.beta = 1.0; 
xa = x00;
restart = data.nv;
k=1;
tol = 1.e-08;



disp('Anderson sith svd')
tic
 fval_aa = [1,problem.cost(xa)];
 pred = problem.prediction(xa);
 acc_aa = [1,problem.accuracy(pred)];

for it = 0 : itmax-1
    xa1 = xa - stepsize * problem.sgd_grad(xa, 100);
    k=k+1;
    [xa, data] = anders1(xa,xa1,data);
    if (mod(it, restart)== 0)
        data.DX = []; data.DF = [];
    end
   if mod(it+1, 10) ==0
       cost = problem.cost(xa);
       fval_aa = [fval_aa;[it+1,cost]];
       pred = problem.prediction(xa);
       test_acc = problem.accuracy(pred);
       acc_aa = [acc_aa;[it+1,test_acc]];
       fprintf(1,' it %d  function val %10.3e, acc is %10.3e \n', it+1, cost, test_acc);
   end
end
toc
t3 = toc;

figure(1)
cyan        = [0.2 0.8 0.8];
brown       = [0.2 0 0];
orange      = [1 0.5 0];
blue        = [0 0.5 1];
green       = [0 0.6 0.3];
red         = [1 0.2 0.2];
purple = [0.4940, 0.1840, 0.5560];

% figure(1)
% start = 5;
% semilogy(fval1(start:end,1), acc1(start:end,2), '-o', 'color',green,'linewidth',2,'MarkerSize',10)
% hold on
% semilogy(fval2(start:end,1), acc2(start:end,2), '-+', 'color',red,'linewidth',2,'MarkerSize',10)
% 
% semilogy(fval_aa(start:end,1), acc_aa(start:end,2), '-s','color',cyan,'linewidth',2,'MarkerSize',10)
% semilogy(fval_aa(start:length(acc_gd),1), acc_gd(start:end), '-','color',brown,'linewidth',2,'MarkerSize',10)
% semilogy(fval_aa(start:length(acc_ncg),1), acc_ncg(start:end), '-*','color',blue,'linewidth',2,'MarkerSize',10)
% semilogy(fval_aa(start:8,1), acc_lb(start:end), '->','color',orange,'linewidth',2,'MarkerSize',10)
% h1=legend('NLTGCR  [1,1]','NLTGCR [10,10]','Anderson table=10','GD-BTK','NCG-BKT','L-BFGS-BKT');
% set(h1,'fontsize',12)
% legend('Location', 'Best')
% hold off
% set(gcf,'paperpositionmode','auto')
% set(gca,'FontSize',12)
% xlabel('Iteration')
% ylabel('Test Accuracy')
% f = gcf;
% exportgraphics(f,'softmax_acc.png','Resolution',300)

start = 1;
interval = 5;
semilogy(fval1(start:end,1), fval1(start:end,2), '-o', 'color',green,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(fval1))
hold on
semilogy(fval2(start:end,1), fval2(start:end,2), '-+', 'color',red,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(fval1))

semilogy(fval_aa(start:end,1), fval_aa(start:end,2), '-s','color',cyan,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(fval1))
load('cost_sgd.mat')
cost_sgdCopy(1) = fval1(1,2);

semilogy(fval_aa(start:length(cost_sgdCopy),1), cost_sgdCopy(start:end), '-','color',brown,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(fval1))
load('cost_ncg.mat')
cost_ncg(1) = fval1(1,2);
semilogy(fval_aa(start:length(cost_ncg),1), cost_ncg(start:end), '-*','color',blue,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(fval1))
ff = load('nltgcr500.mat');
ff = ff.fval1;
ff(1,2) = fval1(1,2);
semilogy(ff(start:length(ff),1), ff(start:end,2), '-d','color',purple,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(fval1))
h1=legend('S-NLTGCR [1,1]','S-NLTGCR [10,10]','S-Anderson table=10','SGD','GD-BTK','NLTGCR [1,1]');
set(h1,'fontsize',12)
legend('Location', 'Best')
hold off
set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',12)
xlabel('Iteration')
ylabel('Log Function Value')
f = gcf;
exportgraphics(f,'softmax_fvalsgd.png','Resolution',300)

