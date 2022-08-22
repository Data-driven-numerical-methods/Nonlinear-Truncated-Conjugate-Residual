function Problem = minimax(A, b,c, solution, stepsize, reg)    
    Problem.name = @() 'minimax bilinear';
    Problem.cost = @cost;
    
    n = length(b);
    AA = [zeros(n)+1e-6*eye(n), stepsize*A; -stepsize*A', zeros(n)+1e-6*eye(n)];
    rhs = [b;-c];
    function res_norm = cost(x)
        res_norm = norm(x - solution);
        res_norm = norm(rhs-AA*x);
    end

    Problem.grad = @grad;
    function fp = grad(x)
         x0 = x(1:n);
         y0 = x(n + 1 : end);
         gx =  A * y0 + b; 
         x1 = x0 - stepsize * gx ;
         gy = A' * x0 + c;
         fp = reg * [gx;-gy] + (1-reg) * [x0;-y0];
    end

    Problem.Agrad = @Agrad;
    function fp = Agrad(x)
        fp = rhs - AA * x;
    end
end

