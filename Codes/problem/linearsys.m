function Problem = linearsys(A, b, stepsize)    
    Problem.name = @() 'linear_system';
    Problem.cost = @cost;
    Problem.sol_opt = A\b;
    function res = cost(x)
        r = b - A*x;
        res = norm(r);
    end
    Problem.grad = @grad;
    function g = grad(x) 
        g = stepsize * (b-A*x);
    end
end

