%% ------ test - Bratu's problem
%---
%--- compare base, linear, adaptive
%---
close all; clear all; clc;
%% ------ init
%--- parameters
maxit = 600;           %--- max num. of iterations
tol = 1e-16;           %--- tol. of stopping criteron
lb = 1;
%--- initialize problem
m = 100;                %--- num. of pts on each row of the grid
alpha = 0;
lambda = 0.5;          %--- lam=10 for indefinite problem
Problem = bratu(m,alpha,lambda);

%% ------ main solve
%x0 = zeros(m*m,1); %--- initial point
x0 = ones(m*m,1);

%%-----------------------------
%%--- method 1: nlTGCR base
%%-----------------------------
fprintf('------ nlTGCR base ------\n')
params = struct('mem_size',lb,'tol',tol,'max_iter',maxit,'restart',inf);
tic
[sol1,fval1,func_eval1] = nltgcr_base(x0,Problem,params);
toc
%%--- result
fval1 = fval1/fval1(1);

%%-----------------------------
%%--- method 2: nlTGCR lin
%%-----------------------------
fprintf('------ nlTGCR lin ------\n')
params = struct('mem_size',lb,'tol',tol,'max_iter',maxit,'restart',inf);
tic
[sol2,fval2,func_eval2] = nltgcr_linear(x0,Problem,params);
toc
%%--- result
fval2 = fval2/fval2(1);

%%-----------------------------
%%--- method 3: nlTGCR ada
%%-----------------------------
fprintf('------ nlTGCR ada ------\n')
params = struct('mem_size',lb,'tol',tol,'max_iter',maxit);
tic
[sol3,fval3,func_eval3] = nltgcr_adaptive(x0,Problem,params);
toc
%%--- aggregate result
fval3 = fval3/fval3(1);

%% ------ plot
%%--- define color
cyan        = [0.2 0.8 0.8];
brown       = [0.8 0.1 0.1];
orange      = [1 0.5 0];
blue        = [0 0.5 1];
green       = [0 0.6 0.3];
red         = [1 0.2 0.2];
black       = [0 0 0];

%%--- figure 1: iteration - relative cost
plot_intv = 20; %--- plot interval
idx1 = [1:plot_intv:size(fval1,1)-1, size(fval1,1)];
idx2 = [1:plot_intv:size(fval2,1)-1, size(fval2,1)];
idx3 = [1:plot_intv:size(fval3,1)-1, size(fval3,1)];


figure(1)
plot(idx1, fval1(idx1), '-o', 'color',blue,'linewidth',2,'MarkerSize',7);
hold on
plot(idx2, fval2(idx2), '--v', 'color',red,'linewidth',2,'MarkerSize',7);
plot(idx3, fval3(idx3), '-.^', 'color',orange,'linewidth',2,'MarkerSize',7);

h1=legend('nlTGCR','nlTGCR linear','nlTGCR adaptive');
set(gca,'YScale','log');
legend('Location', 'southwest', 'FontSize', 14)
hold off
set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',16)
xlabel('Iteration')
ylabel('Cost: ||f(x)||_2/||f(x_0)||_2')
%xlim([0 1001])
ylim([1e-16 1])
%grid on

%%--- figure 2: num. func eval - relative cost
figure(2)
plot(func_eval1(idx1), fval1(idx1), '-o', 'color',blue,'linewidth',2,'MarkerSize',7);
hold on
plot(func_eval2(idx2), fval2(idx2), '--v', 'color',red,'linewidth',2,'MarkerSize',7);
plot(func_eval3(idx3), fval3(idx3), '-.^', 'color',orange,'linewidth',2,'MarkerSize',7);

h1=legend('nlTGCR','nlTGCR linear','nlTGCR adaptive');
set(gca,'YScale','log');
legend('Location', 'southwest', 'FontSize', 14)
hold off
set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',16)
xlabel('Function evaluation')
ylabel('Cost: ||f(x)||_2/||f(x_0)||_2')
%xlim([0 1001])
ylim([1e-16 1])
%grid on