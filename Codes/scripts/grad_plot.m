function grad_plot
spacing = 0.1;
[X,Y] = meshgrid(-2:spacing:2);


%% Rosenbrock%%
% Z = 100*(Y - X.^2)^2 + (1-X).^2;
% DX = -400*(Y-X.^2).*X - 2*(1-X);
% DY = 200*(Y-X.^2);

%% myfun3 %%
% Z = 100*(Y - X.^2)^2 + (1-X).^2;
% DX = -400*(Y-X.^2).*X - 2*(1-X);
% DY = 200*(Y-X.^2);
Z = (X.^2+Y.^2-2)^2;
DX = 4*X.*(X.^2+Y.^2-2) ;     
DY = 4*Y.*(X.^2+Y.^2-2);
normcst = sqrt(DX.^2+DY.^2);
DX = DX./(normcst);
DY = DY./(normcst);
quiver(X,Y,DX,DY)
hold on
contour(X,Y,Z,50)
axis equal
hold off
end