%% ------ test - Lennard-Jones

close all; clear all; clc;
%% ------ init
%--- parameters
maxit = 200;            %--- max num. of iterations
tol = 1.e-16;             %--- tol. of stopping criteron
lb = 10;                 %--- num. of basis vectors to be kept
%--- initialize problem
lattice = 'fcc';       % face-centered cube 
unit_power = 3;        % numb. of cells along 1 axis, N = N_unit * unit_power^3
N_unit = 4;            % must be 4 particles per unit cell for fcc
dim = 3;               % dimensions of the problem, must be 3 for fcc
density = 0.85;        % determines how dense each cell is (affects distance between atoms)

unit_cells = unit_power^3;             
unit_size = (N_unit/density)^(1/3);   
L = unit_size*(unit_power);         % box size      
N = N_unit*unit_cells;              % number of particles    
Problem = LJ(N);


%% ------ main solve
%--- starting point (in flat vector)
load('xinit_pert');
x0 = reshape(x_init,[N*dim,1]);

%%-----------------------------
%%--- method 1: nlTGCR (1)
%%-----------------------------
fprintf('------ nlTGCR (1) ------\n')
params = struct('mem_size',1,'tol',tol,'max_iter',maxit);
tic
[sol1,fval1,func_eval1] = nltgcr_adaptive(x0,Problem,params);
toc

%%-----------------------------
%%--- method 2: nlTGCR (lb)
%%-----------------------------
fprintf('------ nlTGCR (lb) ------\n')
params = struct('mem_size',lb,'tol',tol,'max_iter',maxit);
tic
[sol2,fval2,func_eval2] = nltgcr_adaptive(x0,Problem,params);
toc

%%-----------------------------
%%--- method 3: GD Nesterov
%%-----------------------------
fprintf('------ GD Nesterov ------\n')
params = struct('max_iter',maxit,'verbose',true);
tic
[sol3, infos3] = gd_nesterov(x0,Problem,params);
toc
fval3 = infos3.cost';
func_eval3 = infos3.func_eval';

%%-----------------------------
%%--- method 4: Anderson acceleration
%%-----------------------------
fprintf('------ Anderson ------\n')
params = struct('mem_size',lb,'max_iter',maxit,'tol',tol,'mu',0.001,'beta',0.05,'savsols',false);
tic
[sol4,fval4,~] = anders(x0, Problem, params);
toc
fval4 = fval4';
func_eval4 = [0:length(fval4)-1];

%%-----------------------------
%%--- method 5: Newton Krylov
%%-----------------------------
fprintf('------ Newton-GMRES ------\n')
params = [maxit, 2*lb, 0.9, 2, 1];
tic
[sol5, it_hist5] = nsoli(x0,Problem.grad,Problem.cost,[1.e-16,tol],params);
toc
fval5 = it_hist5(1:end-1,1);
func_eval5 = it_hist5(1:end-1,2);

%%--- aggregate result
fmin = min([fval1;fval2;fval3;fval4;fval5]) - 1.e-16;
sh_fval1 = fval1-fmin;
sh_fval2 = fval2-fmin;
sh_fval3 = fval3-fmin;
sh_fval4 = fval4-fmin;
sh_fval5 = fval5-fmin;

%% ------ plot
%%--- define color
cyan        = [0.2 0.8 0.8];
brown       = [0.8 0.1 0.1];
orange      = [1 0.5 0];
blue        = [0 0.5 1];
green       = [0 0.6 0.3];
red         = [1 0.2 0.2];

%%--- avg and std err
plot_intv = 10; %--- plot interval
idx1 = [1:plot_intv:size(fval1,1)-1, size(fval1,1)];
idx2 = [1:plot_intv:size(fval2,1)-1, size(fval2,1)];
idx3 = [1:plot_intv:size(fval3,1)-1, size(fval3,1)];
idx4 = [1:plot_intv:size(fval4,1)-1, size(fval4,1)];

%%--- figure 1: num. func eval - relative cost
figure(1)
plot(func_eval1(idx1), sh_fval1(idx1), '-o', 'color',blue,'linewidth',2,'MarkerSize',7);
hold on
plot(func_eval2(idx2), sh_fval2(idx2), '-v', 'color',green,'linewidth',2,'MarkerSize',7);
plot(func_eval3(idx3), sh_fval3(idx3), '-.+', 'color',red,'linewidth',2,'MarkerSize',7);
plot(func_eval4(idx4), sh_fval4(idx4), '--^', 'color',cyan,'linewidth',2,'MarkerSize',7);
plot(func_eval5, sh_fval5, '-.s', 'color',brown,'linewidth',2,'MarkerSize',7);

h1=legend('nlTGCR (m=1)', ...
    ['nlTGCR (m=' num2str(lb) ')'], ...
    'Nesterov', ...
    'AA', ...
    'Newton-GMRES');
set(gca,'YScale','log');
legend('Location', 'southwest', 'FontSize', 14)
hold off
set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',16)
xlabel('Function evaluation')
ylabel('Cost: E - E_m_i_n')
ylim([1e-13 1e4])
xlim([0 220])

%%--- figure 2: 3D illustration
figure(2)
coor1 = reshape(x0,dim,N);
coor2 = reshape(sol1,dim,N);

scatter3(coor1(1,:),coor1(2,:),coor1(3,:),'b','filled')
hold on
scatter3(coor2(1,:),coor2(2,:),coor2(3,:),'r^','LineWidth',2,'MarkerFaceColor','r')
for i = 1:N
    plot3([coor1(1,i) coor2(1,i)],[coor1(2,i) coor2(2,i)],[coor1(3,i) coor2(3,i)],'k:','LineWidth',1.5)
end

h1=legend('Initial position', 'Final position');
legend('Location', 'northwest', 'FontSize', 14)
hold off