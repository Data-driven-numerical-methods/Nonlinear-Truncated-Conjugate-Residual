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

v0 = [];v2 = [];v5 = [];v7 = [];
av0 = [];av2=[];av5=[];av7=[];
for e = 1:3  
    % Define Problem
    problem = softmax_regression(x_train, y_train, x_test, y_test, lambda, l);
		        
    
    x00 = randn(d*l,1);
    
    stepsize = 1;
    % Define FF
    F = @(x) stepsize * problem.sgd_grad(x, 500);
    %%-------------------- set some params
    itmax = 400;
    tol = 1.e-09;
    x0  = x00;
    x2 = x00;
    x3 = x00;
    k= 0;
    epsf = 1e-2;
    disp('testing nltgcr')
    tol = 1.e-06; 
    lb = 1; restart = 20;
    
    [sol1, fval1, acc1, ~, ~]= nltgcr3(F,x0,lb,tol,itmax, problem, restart, epsf,0);
    av0 = [av0,acc1];
    v0 = [v0,fval1];
    
    %%-------------------- nltgcr2 
    disp('testing nltgcr')
    lb = 1; restart = 20;
    
    [~, fval2, acc2, ~, ~]= nltgcr3(F,x2,lb,tol,itmax, problem, restart, epsf,0.2);
    
    av2 = [av2,acc2];
    v2 = [v2,fval2];
    
    disp('testing nltgcr')
    lb = 1; restart = 20;
    
    [~, fval3, acc3, ~, ~]= nltgcr3(F,x3,lb,tol,itmax, problem, restart, epsf,0.5);
    av5 = [av5,acc3];
    v5 = [v5,fval3];
    
    disp('testing nltgcr')
    lb = 1; restart = 10;
    
    [~, fval4, acc4, ~, ~]= nltgcr3(F,x3,lb,tol,itmax, problem, restart, epsf,0.7);
    av7 = [av7,acc4];
    v7 = [v7,fval4];
end


meanV0 = mean(v0(:,2:2:end),2);
errV0 = std(v0(:,2:2:end),[],2);
meanV2 = mean(v2(:,2:2:end),2);
errV2 = std(v2(:,2:2:end),[],2);

meanV5 = mean(v5(:,2:2:end),2);
errV5 = std(v5(:,2:2:end),[],2);
meanV7 = mean(v7(:,2:2:end),2);
errV7 = std(v7(:,2:2:end),[],2);

figure(1)
cyan        = [0.2 0.8 0.8];
brown       = [0.2 0 0];
orange      = [1 0.5 0];
blue        = [0 0.5 1];
green       = [0 0.6 0.3];
red         = [1 0.2 0.2];


figure(1)
start = 6;
x = linspace(0,itmax,41);
errorbar(x(start:end), meanV0(start:end), errV0(start:end), '-o', 'color',blue,'linewidth',2,'MarkerSize',7);
hold on
errorbar(x(start:end),meanV2(start:end),errV2(start:end), '-+', 'color',red,'linewidth',2,'MarkerSize',7);
errorbar(x(start:end), meanV5(start:end),errV5(start:end), '--s', 'color',green,'linewidth',2,'MarkerSize',7);
errorbar(x(start:end), meanV7(start:end),errV7(start:end), '-*', 'color',orange,'linewidth',2,'MarkerSize',7);
h1=legend('S-NLTGCR, v=0', 'S-NLTGCR, v=0.2', 'S-NLTGCR, v=0.5','S-NLTGCR, v=0.7');
set(h1,'fontsize',12)
set(gca,'YScale','log');
legend('Location', 'Best')
hold off
set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',12)
xlabel('Iteration')
ylabel('Loss')
f = gcf;
exportgraphics(f,'softmax_loss_mom.png','Resolution',300)

figure(2)
ameanV0 = mean(av0(:,2:2:end),2);
aerrV0 = std(av0(:,2:2:end),[],2);
ameanV2 = mean(av2(:,2:2:end),2);
aerrV2 = std(av2(:,2:2:end),[],2);

ameanV5 = mean(av5(:,2:2:end),2);
aerrV5 = std(av5(:,2:2:end),[],2);
ameanV7 = mean(av7(:,2:2:end),2);
aerrV7 = std(av7(:,2:2:end),[],2);


errorbar(x(start:end), ameanV0(start:end),aerrV0(start:end), '-o', 'color',blue,'linewidth',2,'MarkerSize',7);
hold on
errorbar(x(start:end), ameanV2(start:end),aerrV2(start:end), '-+', 'color',red,'linewidth',2,'MarkerSize',7);
errorbar(x(start:end), ameanV5(start:end),aerrV5(start:end), '--s', 'color',green,'linewidth',2,'MarkerSize',7);
errorbar(x(start:end), ameanV7(start:end),aerrV7(start:end), '-*', 'color',orange,'linewidth',2,'MarkerSize',7);
h1=legend('S-NLTGCR, v=0', 'S-NLTGCR, v=0.2', 'S-NLTGCR, v=0.5','S-NLTGCR, v=0.7');
set(h1,'fontsize',12)
legend('Location', 'Best')
hold off
set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',12)
xlabel('Iteration')
ylabel('Test Accuracy')
f = gcf;
exportgraphics(f,'softmax_mom_ACC.png','Resolution',300)


