clear all
fun = @rosenbrockwithgrad2;
x0 = [-1.5; 1];

options = optimoptions('fminunc','Algorithm','quasi-newton','SpecifyObjectiveGradient',true);
options.Display = 'iter';
[x, fval, exitflag, output] = fminunc(fun,x0,options);
sol = x;
% fun = @rosenbrockwithgrad; sol = ones(2,1);


lb = 2; tol = 1e-6; itmax = 50; restart = 5;

[x_2,histout,costdata, history] = ntrust_CTK(x0,fun,tol,itmax,1d-14);


[x_3,FVAL, history2] = nltgcr_opt(fun,x0,lb,tol,itmax, restart);
alg_list = {'ntrust','nlTGCR'};
iter_hist = {history,history2 }; fval_hist = {histout(:,2),FVAL};
draw_convergence_sequence(fun,sol, alg_list,iter_hist,fval_hist);

figure(2)
semilogy(histout(:,2),'r');
hold on 
semilogy(FVAL,'b');
