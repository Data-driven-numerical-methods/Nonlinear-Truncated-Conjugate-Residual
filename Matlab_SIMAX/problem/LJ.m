function Problem = LJ(nat)
%% Lennard-Jones
%  vector form - flat input flat output
%  -  energy and forces for Lennard Jones potential
%  -  input: nat: number of atoms
%  -         rxyz: positions of atoms in vector format
%  -  output: etot: total energy
%  -          fxyz: forces (negative derivative of energy with
%             respect to positions)
% %         dimension rxyz(3*nat,1),fxyz(3*nat,1)
%  THIS RETURNES GRADIENT INSTEAD OF NEGATIVE OF GRADIENT    

    %---------- parameters
    dim = 3;
    nn = dim*nat;
    
    %---------- problem info
    Problem.name = @() 'Lennard-Jones';
    Problem.dim = @() nn;
    Problem.samples = @() 0;
     
    %---------- cost
    Problem.cost = @cost;
    function c = cost(u)
        [c,~] = LJ_potV(u);
    end
    
    %---------- gradient
    Problem.grad = @grad;
    function g = grad(u)
        [~,g] = LJ_potV(u);
    end
    
    %---------- feval
    Problem.feval = @LJ_potV;
    function [etot, fxyz] = LJ_potV(rxyz)
        %---------- initialize etot and forces
        etot =0.0;
        fxyz = zeros(nn,1);
        %----------  atom loop
        for iat=1:nat
            i1 = dim*(iat-1)+1;
            for jat=1:iat-1
                j1 = dim*(jat-1)+1;
                dvec = rxyz(i1:i1+2)-rxyz(j1:j1+2);
                %---------- distance^2     
                dd = transpose(dvec)*dvec;
                di = 1.0/dd;
                ri6 = di^3;
                ri12 = ri6*ri6;
                %---------- dimensionless Lennard-Jones
                etot = etot + 4.0*(ri12-ri6);
                %---------- forces: + SIGN; gradient: - SIGN
                tt= -24.0*di*(2.0*ri12-ri6);
                %---------- GRADIENT
                dvec = tt*dvec ;
                fxyz(i1:i1+2) = fxyz(i1:i1+2) + dvec;
                fxyz(j1:j1+2) = fxyz(j1:j1+2) - dvec;
            end
        end
    end
end
 
