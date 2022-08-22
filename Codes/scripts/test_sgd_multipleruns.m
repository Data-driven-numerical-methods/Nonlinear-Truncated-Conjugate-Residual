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
B1000 = [];
B500 =[];
AA500 = [];
AA1000 = [];
accAA5 = [];
accAA10 = [];
accT5 = [];
accT10 = [];
for e = 1:4
% Define Problem
    problem = softmax_regression(x_train, y_train, x_test, y_test, lambda, l);
    x00 = randn(d*l,1);
    stepsize = 1;
    % Define FF
    
    %%-------------------- set some params
    itmax = 200;
    tol = 1.e-09;
    x0  = x00;
    k= 0;
    
    disp('testing nltgcr')
    tol = 1.e-06; 
    lb = 1; 
    restart = 10;
    %%-------------------- nltgcr2 
    epsf = 1;
    F = @(x) stepsize * problem.sgd_grad(x, 1000);
    [sol1, fval1, acc1, ~, ~]= nltgcr3(F,x00,lb,tol,itmax, problem, restart, epsf,0);
    accT5 = [accT5, acc1];
    
    B500 = [B500, fval1];
    %%-------------------- nltgcr2 
    disp('testing nltgcr')
    lb = 10; 
    restart = 10;
    F = @(x) stepsize * problem.sgd_grad(x, 1000);
    [~, fval2, acc2, ~, ~]= nltgcr3(F,x00,lb,tol,itmax, problem, restart, epsf,0.0);
    B1000 = [B1000, fval2];
    accT10 = [accT10, acc1];


    %%-------------------- 
    %% cost3: Anderson With SVD
    data.DX = [];  data.DF = []; data.nv = 1; data.beta = 1.0; 
    xa = x00;
    restart = data.nv;
    k=1;
    tol = 1.e-08;
    
    disp('Anderson sith svd')
    fval_aa = [1,problem.cost(xa)];
    pred = problem.prediction(xa);
    acc_aa = [1,problem.accuracy(pred)];
    
    for it = 0 : itmax-1
        xa1 = xa - stepsize * problem.sgd_grad(xa, 1000);
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
    AA500 = [AA500,fval_aa];
    accAA5 = [accAA5,acc_aa];

    data.DX = [];  data.DF = []; data.nv = 10; data.beta = 1.0; 
    xa = x00;
    restart = data.nv;
    k=1;
    tol = 1.e-08;
    
    disp('Anderson sith svd')

     fval_aa2 = [1,problem.cost(xa)];
     pred = problem.prediction(xa);
     acc_aa2 = [1,problem.accuracy(pred)];
    
    for it = 0 : itmax-1
        xa1 = xa - stepsize * problem.sgd_grad(xa, 1000);
        k=k+1;
        [xa, data] = anders1(xa,xa1,data);
        if (mod(it, restart)== 0)
            data.DX = []; data.DF = [];
        end
       if mod(it+1, 10) ==0
           cost = problem.cost(xa);
           fval_aa2 = [fval_aa2;[it+1,cost]];
           pred = problem.prediction(xa);
           test_acc = problem.accuracy(pred);
           acc_aa2 = [acc_aa2;[it+1,test_acc]];
           fprintf(1,' it %d  function val %10.3e, acc is %10.3e \n', it+1, cost, test_acc);
       end
    end
    AA1000 = [AA1000,fval_aa2];
    accAA10 = [accAA10,acc_aa2];
end
figure(1)
cyan        = [0.2 0.8 0.8];
brown       = [0.2 0 0];
orange      = [1 0.5 0];
blue        = [0 0.5 1];
green       = [0 0.6 0.3];
red         = [1 0.2 0.2];
purple = [0.4940, 0.1840, 0.5560];

meanAA500 = mean(AA500(:,2:2:end),2);
errAA500 = std(AA500(:,2:2:end),[],2);
meanAA1000 = mean(AA1000(:,2:2:end),2);
errAA1000 = std(AA1000(:,2:2:end),[],2);

meanB500 = mean(B500(:,2:2:end),2);
errB500 = std(B500(:,2:2:end),[],2);
meanB1000 = mean(B1000(:,2:2:end),2);
errB1000 = std(B1000(:,2:2:end),[],2);
figure(1)
x = linspace(0,200,21);
errorbar(x, meanB500,errB500, '-o', 'color',blue,'linewidth',2,'MarkerSize',7);
hold on
errorbar(x,meanB1000,errB1000, '-+', 'color',red,'linewidth',2,'MarkerSize',7);
errorbar(x, meanAA500,errAA500, '--s', 'color',green,'linewidth',2,'MarkerSize',7);
errorbar(x, meanAA1000,errAA1000, '-*', 'color',orange,'linewidth',2,'MarkerSize',7);
set(gca,'YScale','log');
legend({'S-NLTGCR m=1', 'S-NLTGCR m=10', 'S-Anderson m=1','S-Anderson m=10'},'Location', 'Best')
hold off

set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',12)
xlabel('Iteration')
ylabel('Log Function Value')
f = gcf;
exportgraphics(f,'softmax_sloss_avg2.png','Resolution',300)


ameanAA500 = mean(accAA5(:,2:2:end),2);
aerrAA500 = std(accAA5(:,2:2:end),[],2);
ameanAA1000 = mean(accAA10(:,2:2:end),2);
aerrAA1000 = std(accAA10(:,2:2:end),[],2);


ameanB500 = mean(accT5(:,2:2:end),2);
aerrB500 = std(accT5(:,2:2:end),[],2);
ameanB1000 = mean(accT10(:,2:2:end),2);
aerrB1000 = std(accT10(:,2:2:end),[],2);
figure(2)
start = 1;
interval = 2;
errorbar(x, ameanB500(start:end),aerrB500(start:end), '-o', 'color',blue,'linewidth',2,'MarkerSize',7);
hold on
errorbar(x, ameanB1000(start:end),aerrB1000(start:end), '-+', 'color',red,'linewidth',2,'MarkerSize',7);
errorbar(x, ameanAA500(start:end),aerrAA500(start:end), '--s', 'color',green,'linewidth',2,'MarkerSize',7);
errorbar(x, ameanAA1000(start:end),aerrAA1000(start:end), '-*', 'color',orange,'linewidth',2,'MarkerSize',7);
legend({'S-NLTGCR m=1', 'S-NLTGCR m=10', 'S-Anderson m=1','S-Anderson m=10'},'Location', 'Best')
hold off
set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',12)
xlabel('Iteration')
ylabel('Test Accuracy')
f = gcf;
ylim([0.6,0.8]);
exportgraphics(f,'softmax_sacc_avg2.png','Resolution',300)

