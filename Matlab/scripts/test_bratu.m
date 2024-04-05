%% ------ test - Bratu's problem
%---
%--- Problem:
%---
%---    PDE:
%---        u_xx + u_yy + alpha * u_x + lambda * exp(u) = 0
%---        with Dirichlet boundary conditions:
%---        u(x,y) = 0 on unit square domain
%---
%---    Finite Difference Method on Bratu's Problem
%---        D * u + alpha * J * u + lambda * exp(u) = 0
%---
close all; clear all; clc;
%% ------ init
%--- parameters 
maxit = 300;            %--- max num. of iterations
tol = 1e-16;            %--- tol. of stopping criteron
lb = 10;                %--- num. of basis vectors to be kept
%--- initialize problem
m = 100;                %--- num. of pts on each row of the grid
alpha = 0;
lambda = 0.5;
Problem = bratu(m,alpha,lambda);


%% ------ main solve
x0 = zeros(m*m,1); %--- initial point
%x0 = ones(m*m,1);
f0 = Problem.cost(x0);

%%-----------------------------
%%--- method 1: nlTGCR
%%-----------------------------
fprintf('------ nlTGCR ------\n')
params = struct('mem_size',1,'tol',tol,'max_iter',maxit);
tic
[sol1,fval1,func_eval1] = nltgcr_adaptive(x0,Problem,params);
toc
%%--- result
fval1 = fval1/fval1(1);

%%-----------------------------
%%--- method 2: GD Nesterov
%%-----------------------------
fprintf('------ GD Nesterov ------\n')
params = struct('max_iter',maxit,'f_opt',0,'verbose',true);
tic
[sol2, infos2] = gd_nesterov(x0,Problem,params);
toc
%%--- result
fval2 = infos2.cost';
func_eval2 = infos2.func_eval';
fval2 = fval2/fval2(1);

%%-----------------------------
%%--- method 3: L-BFGS
%%-----------------------------
fprintf('------ L-BFGS ------\n')
params = struct('mem_size',lb,'max_iter',maxit,'f_opt',0,'verbose',true);
tic
[sol3, infos3] = lbfgs(x0, Problem, params);
fval3 = infos3.cost';
func_eval3 = infos3.func_eval';
toc
%%--- result
fval3 = fval3/fval3(1);

%%-----------------------------
%%--- method 4: Anderson acceleration
%%-----------------------------
fprintf('------ Anderson ------\n')
params = struct('mem_size',lb,'max_iter',maxit,'tol',tol,'mu',1.0,'beta',0.1,'savsols',false);
tic
[sol4,fval4,~] = anders(x0, Problem, params);
toc
%%--- result
fval4 = fval4'/fval4(1);
func_eval4 = [0:length(fval4)-1]';

%%-----------------------------
%%--- method 5: nonlinear CG
%%-----------------------------
fprintf('------ nonlinear CG ------\n')
params = struct('max_iter',maxit,'f_opt',0,'verbose',true);
tic
[sol5, infos5] = ncg(x0,Problem,params);
fval5 = infos5.cost';
func_eval5 = infos5.func_eval';
toc
%%--- result
fval5 = fval5/fval5(1);

%%-----------------------------
%%--- method 6: Newton CG
%%-----------------------------
fprintf('------ Newton-CG ------\n')
params = [maxit, 5*lb, 0.9, 6, 0]; %--- restart is max steps for inner loop
tic
[sol6, it_hist6] = nsoli(x0,Problem.grad,Problem.cost,[tol,tol],params);
toc
%%--- result
fval6 = it_hist6(1:end-1,1)/it_hist6(1,1);
func_eval6 = it_hist6(1:end-1,2);


%% ------ plot
%%--- define color
cyan        = [0.2 0.8 0.8];
brown       = [0.8 0.1 0.1];
orange      = [1 0.5 0];
blue        = [0 0.5 1];
green       = [0 0.6 0.3];
red         = [1 0.2 0.2];

%%--- figure: num. func eval - relative cost
figure(1)
plot(func_eval1, fval1, '-o', 'color',blue,'linewidth',2,'MarkerSize',7,'MarkerIndices',[1:10:size(fval1,1)-1, size(fval1,1)]);
hold on
plot(func_eval2, fval2, '--+', 'color',red,'linewidth',2,'MarkerSize',7,'MarkerIndices',[1:10:size(fval2,1)-1, size(fval2,1)]);
plot(func_eval3, fval3, '-.d', 'color',orange,'linewidth',2,'MarkerSize',7,'MarkerIndices',[1:10:size(fval3,1)-1, size(fval3,1)]);
plot(func_eval4, fval4, '--^', 'color',cyan,'linewidth',2,'MarkerSize',7,'MarkerIndices',[1:10:size(fval4,1)-1, size(fval4,1)]);
plot(func_eval5, fval5, '-v', 'color',green,'linewidth',2,'MarkerSize',7,'MarkerIndices',[1:10:size(fval5,1)-1, size(fval5,1)]);
plot(func_eval6, fval6, '-.s', 'color',brown,'linewidth',2,'MarkerSize',7);

h1=legend('nlTGCR (m=1)', ...
    'Nesterov', ...
    'L-BFGS', ...
    'AA', ...
    'NCG', ...
    'Newton-CG');
set(gca,'YScale','log');
legend('Location', 'southwest', 'FontSize', 14)
hold off
set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',16)
xlabel('Function evaluation')
ylabel('Cost: ||f(x)||_2/||f(x_0)||_2')
xlim([0 300])