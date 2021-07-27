clear; close all; clc;

N_CASE = 6;
h = zeros(1, N_CASE);
err = zeros(3, N_CASE);

for i = 1:N_CASE
    N = 2^i;
    [ch, ~, ce] = solve_2d_elliptic_pde(N, N);
    h(i) = ch;
    err(1,i) = ce(1);
    err(2,i) = ce(2);
    err(3,i) = ce(3);
    fprintf("h=1/%d, |err|_inf=%e, |err|_L2=%e, |err|_H1=%e\n", N, err(1,i), err(2,i), err(3,i));
end

loglog(h, err(1,:), '-s')
hold on
loglog(h, err(2,:), '-s')
hold on
loglog(h, err(3,:), '-s')
grid on
legend('inf','L2','H1','Location','northwest')

function [h1, h2, errnorm] = solve_2d_elliptic_pde(N1, N2)
    global P T Pb Tb Jac

    x_min = 0.0; x_max = 1.0; h1 = (x_max-x_min)/N1;
    y_min = 0.0; y_max = 1.0; h2 = (y_max-y_min)/N2;
    
    [P, T] = mesh_info_mat(x_min,x_max,y_min,y_max,N1,N2);
    [Pb, Tb] = fem_info_mat(x_min,x_max,y_min,y_max,N1,N2);
    [boundary_edge, boundary_node] = boundary_info_mat(N1, N2, T, Tb);
    
    Nlb = size(Tb, 1); % Num of local basis functions
    Nb = size(Pb, 2); % Num of global basis functions(fem unknowns)
    N = size(T,2); % Num of mesh/fem elements
    Nm = size(P, 2); % Num of mesh nodes
    nbn = size(boundary_node, 2);
    
    Jac = zeros(N, 1);
    for i = 1:N
        p1 = P(:, T(1, i));
        p2 = P(:, T(2, i));
        p3 = P(:, T(3, i));
        Jac(i) = calc_elem_jacobi(p1, p2, p3);
    end
    
    gauss_quad_x0=[1.0/3, 1.0/5, 3.0/5, 1.0/5];
    gauss_quad_y0=[1.0/3, 1.0/5, 1.0/5, 3.0/5];
    gauss_quad_w=[-27.0/96, 25.0/96, 25.0/96, 25.0/96];
    gauss_quad_n=4;
    
    % Assemble the stiffness matrix
    A = sparse(Nb, Nb);
    for n = 1:N
        for alpha = 1:Nlb % trial
            for beta = 1:Nlb % test
                i = Tb(beta, n);
                j = Tb(alpha, n);
                
                r = 0.0;
                for k = 1:gauss_quad_n
                    [x_loc, y_loc] = affine_mapping_back(n, gauss_quad_x0(k), gauss_quad_y0(k));
                    tmp1 = grad_trial(alpha, n, x_loc, y_loc);
                    tmp2 = grad_test(beta, n, x_loc, y_loc);
                    tmp3 = c(x_loc, y_loc) * (tmp1(1)*tmp2(1) + tmp1(2)*tmp2(2));
                    r = r + gauss_quad_w(k) * tmp3;
                end
                r = r * abs(Jac(n));
                
                A(i, j) = A(i, j) + r;
            end
        end
    end
    
    % Assemble the load vector
    b = zeros(Nb, 1);
    for n = 1:N
        for beta = 1:Nlb % test
            i = Tb(beta, n);
            
            r = 0.0;
            for k = 1:gauss_quad_n
                [x_loc, y_loc] = affine_mapping_back(n, gauss_quad_x0(k), gauss_quad_y0(k));
                r = r + gauss_quad_w(k) * f(x_loc, y_loc) * test(beta, n, x_loc, y_loc);
            end
            r = r * abs(Jac(n));
            b(i) = b(i) + r;
        end
    end
    
    % Boundary
    for k = 1:nbn
        if boundary_node(1, k) == -1
            i = boundary_node(2, k);
            A(i, :) = 0;
            A(i, i) = 1;
            b(i) = u(Pb(1, i), Pb(2, i));
        end
    end
    
    % Solve and Check
    u_sol = A\b;
    
    errnorm = zeros(1, 3); %inf, L2, H1 respectively
    for n = 1:N
        for k = 1:gauss_quad_n
            x0 = gauss_quad_x0(k);
            y0 = gauss_quad_y0(k);
            
            [x, y] = affine_mapping_back(n, x0, y0);
            
            w = 0.0;
            for i = 1:Nlb
                w = w + u_sol(Tb(i, n)) * trial_ref(i, x0, y0);
            end
            
            err = abs(w - u(x, y));
            if err > errnorm(1)
                errnorm(1) = err;
            end
        end
    end
    
    for n = 1:N
        res = 0.0;
        for k = 1:gauss_quad_n
            x0 = gauss_quad_x0(k);
            y0 = gauss_quad_y0(k);
            
            [x, y] = affine_mapping_back(n, x0, y0);
            
            w = 0.0;
            for i = 1:Nlb
                w = w + u_sol(Tb(i, n)) * trial_ref(i, x0, y0);
            end
            
            err = (w - u(x, y))^2;
            res = res + gauss_quad_w(k) * err;
        end
        res = res * abs(Jac(n));
        errnorm(2) = errnorm(2) + res;
    end
    errnorm(2) = sqrt(errnorm(2));
    
    for n = 1:N
        res = 0.0;
        for k = 1:gauss_quad_n
            x0 = gauss_quad_x0(k);
            y0 = gauss_quad_y0(k);
            
            [x, y] = affine_mapping_back(n, x0, y0);
            
            w = zeros(2, 1);
            for i = 1:Nlb
                w = w + u_sol(Tb(i, n)) * grad_trial(i, n, x, y);
            end
            
            err = norm(w - grad_u(x, y))^2;
            res = res + gauss_quad_w(k) * err;
        end
        res = res * abs(Jac(n));
        errnorm(3) = errnorm(3) + res;
    end
    errnorm(3) = sqrt(errnorm(3));
end

function [P, T] = mesh_info_mat(xmin, xmax, ymin, ymax, n1, n2)
    h1 = (xmax-xmin)/n1;
    h2 = (ymax-ymin)/n2;
    
    P = zeros(2, (n1+1)*(n2+1));
    T = zeros(3, 2*n1*n2);
    
    node_idx = zeros(n1+1, n2+1);
    
    for i = 1:n1+1
        x = xmin + (i-1)*h1;
        for j = 1:n2+1
            y = ymin + (j-1)*h2;
            idx = (i-1)*(n2+1)+j;
            P(:,idx) = [x, y];
            node_idx(i, j) = idx;
        end
    end
    
    for i = 1:n1
        for j = 1:n2
            quad_idx = j + (i-1)*n2;
            tri_idx0 = 2*quad_idx-1;
            tri_idx1 = 2*quad_idx;
            idx = [node_idx(i, j), node_idx(i+1, j), node_idx(i+1, j+1), node_idx(i, j+1)];
            T(:,tri_idx0) = [idx(1),idx(2),idx(4)];
            T(:,tri_idx1) = [idx(4),idx(2),idx(3)];
        end
    end
end

function [Pb, Tb] = fem_info_mat(xmin, xmax, ymin, ymax, n1, n2)
    half_h1 = (xmax-xmin)/n1/2;
    half_h2 = (ymax-ymin)/n2/2;

    node_num = (2*n1+1)*(2*n2+1);
    elem_num = 2*n1*n2;
    
    Pb = zeros(2, node_num);
    Tb = zeros(6, elem_num);
    
    node_idx = zeros(2*n1+1, 2*n2+1);
    
    for i = 1:2*n1+1
        x = xmin + (i-1)*half_h1;
        for j = 1:2*n2+1
            y = ymin + (j-1)*half_h2;
            idx = j + (i-1)*(2*n2+1);
            Pb(:, idx) = [x, y];
            node_idx(i, j) = idx;
        end
    end
    
    for i = 1:n1
        for j = 1:n2
            quad_idx = j + (i-1)*n2;
            tri_idx0 = 2*quad_idx-1;
            tri_idx1 = 2*quad_idx;
            
            i0 = 2*i-1;
            j0 = 2*j-1;
            idx = zeros(1, 9);
            idx(1) = node_idx(i0, j0);
            idx(2) = node_idx(i0+1, j0);
            idx(3) = node_idx(i0+2, j0);
            idx(4) = node_idx(i0, j0+1);
            idx(5) = node_idx(i0+1, j0+1);
            idx(6) = node_idx(i0+2, j0+1);
            idx(7) = node_idx(i0, j0+2);
            idx(8) = node_idx(i0+1, j0+2);
            idx(9) = node_idx(i0+2, j0+2);
            
            Tb(:,tri_idx0) = [idx(1),idx(3),idx(7),idx(2),idx(5),idx(4)];
            Tb(:,tri_idx1) = [idx(7),idx(3),idx(9),idx(5),idx(6),idx(8)];
        end
    end
end

function [bdry_edge, bdry_node] = boundary_info_mat(n1, n2, T, Tb)
    bdry_edge = zeros(4, 2*(n1+n2));
    bdry_node = zeros(2, 4*(n1+n2));
    
    % Bottom
    for k = 1:n1
        edge_idx = k;
        elem_idx = 1 + (k-1)*n2*2;
        node_idx = 2*edge_idx-1;

        bdry_edge(1, edge_idx) = -1;
        bdry_edge(2, edge_idx) = elem_idx;
        bdry_edge(3, edge_idx) = T(1, elem_idx);
        bdry_edge(4, edge_idx) = T(2, elem_idx);
        
        bdry_node(1, node_idx) = -1;
        bdry_node(2, node_idx) = Tb(1, elem_idx);
        
        bdry_node(1, node_idx+1) = -1;
        bdry_node(2, node_idx+1) = Tb(4, elem_idx);
    end
    
    % Right
    for k = 1:n2
        edge_idx = k+n1;
        elem_idx = 2*n2*(n1-1) + 2*k;
        node_idx = 2*edge_idx-1;

        bdry_edge(1, edge_idx) = -1;
        bdry_edge(2, edge_idx) = elem_idx;
        bdry_edge(3, edge_idx) = T(2, elem_idx);
        bdry_edge(4, edge_idx) = T(3, elem_idx);
        
        bdry_node(1, node_idx) = -1;
        bdry_node(2, node_idx) = Tb(2, elem_idx);
        
        bdry_node(1, node_idx+1) = -1;
        bdry_node(2, node_idx+1) = Tb(5, elem_idx);
    end
    
    % Top
    for k = 1:n1
        edge_idx = k+n2+n1;
        elem_idx = 2*n1*n2 - 2*n2*(k-1);
        node_idx = 2*edge_idx-1;
        
        bdry_edge(1, edge_idx) = -1;
        bdry_edge(2, edge_idx) = elem_idx;
        bdry_edge(3, edge_idx) = T(3, elem_idx);
        bdry_edge(4, edge_idx) = T(1, elem_idx);
        
        bdry_node(1, node_idx) = -1;
        bdry_node(2, node_idx) = Tb(3, elem_idx);
        
        bdry_node(1, node_idx+1) = -1;
        bdry_node(2, node_idx+1) = Tb(6, elem_idx);
    end
    
    % Left
    for k = 1:n2
        edge_idx = k+2*n1+n2;
        elem_idx = 2*n2 - (2*k-1);
        node_idx = 2*edge_idx-1;

        bdry_edge(1, edge_idx) = -1;
        bdry_edge(2, edge_idx) = elem_idx;
        bdry_edge(3, edge_idx) = T(3, elem_idx);
        bdry_edge(4, edge_idx) = T(1, elem_idx);
        
        bdry_node(1, node_idx) = -1;
        bdry_node(2, node_idx) = Tb(3, elem_idx);
        
        bdry_node(1, node_idx+1) = -1;
        bdry_node(2, node_idx+1) = Tb(6, elem_idx);
    end
end

function [x, y] = affine_mapping_back(n, x0, y0)
    global P T
    
    P1 = P(:, T(1, n));
    P2 = P(:, T(2, n));
    P3 = P(:, T(3, n));

    x1 = P1(1); y1 = P1(2);
    x2 = P2(1); y2 = P2(2);
    x3 = P3(1); y3 = P3(2);
    
    x = (x2-x1)*x0 + (x3-x1)*y0 + x1;
    y = (y2-y1)*x0 + (y3-y1)*y0 + y1;
end

function [x0, y0] = affine_mapping(n, x, y)
    global P T Jac
    
    P1 = P(:, T(1, n));
    P2 = P(:, T(2, n));
    P3 = P(:, T(3, n));
    
    x1 = P1(1); y1 = P1(2);
    x2 = P2(1); y2 = P2(2);
    x3 = P3(1); y3 = P3(2);
    
    x0 = ((y3-y1)*(x-x1)-(x3-x1)*(y-y1))/Jac(n);
    y0 = -((y2-y1)*(x-x1)-(x2-x1)*(y-y1))/Jac(n);
end

function [J] = calc_elem_jacobi(P1, P2, P3)
    x1 = P1(1); y1 = P1(2);
    x2 = P2(1); y2 = P2(2);
    x3 = P3(1); y3 = P3(2);
    J = (x2-x1)*(y3-y1)-(x3-x1)*(y2-y1);
end

function [ret] = grad_trial(basis, n, x, y)
    ret = grad_test(basis, n, x, y);
end

function [ret] = grad_test(basis, n, x, y)
    global P T Jac

    [x0, y0] = affine_mapping(n, x, y);
    gp = grad_test_ref(basis, x0, y0);
    
    dpx0 = gp(1);
    dpy0 = gp(2);
    
    P1 = P(:, T(1, n));
    P2 = P(:, T(2, n));
    P3 = P(:, T(3, n));
    
    x1 = P1(1); y1 = P1(2);
    x2 = P2(1); y2 = P2(2);
    x3 = P3(1); y3 = P3(2);

    ret = [dpx0 * (y3-y1) + dpy0 * (y1-y2); dpx0 * (x1-x3) + dpy0 * (x2-x1)] / Jac(n);
end

function [ret] = grad_trial_ref(basis, x0, y0)
    ret = grad_test_ref(basis, x0, y0);
end

function [ret] = grad_test_ref(basis, x0, y0)
    switch(basis)
        case 1
            ret = [4*x0+4*y0-3; 4*y0+4*x0-3];
        case 2
            ret = [4*x0-1; 0];
        case 3
            ret = [0; 4*y0-1];
        case 4
            ret = [-8*x0-4*y0+4; -4*x0];
        case 5
            ret = [4*y0; 4*x0];
        case 6
            ret = [-4*y0; -8*y0-4*x0+4];
        otherwise
            ret = [0; 0];
    end
end

function [ret] = trial(basis, n, x, y)
    ret = test(basis, n, x, y);
end

function [ret] = test(basis, n, x, y)
    [x0, y0] = affine_mapping(n, x, y);
    ret = test_ref(basis, x0, y0);
end

function [ret] = trial_ref(basis, x0, y0)
    ret = test_ref(basis, x0, y0);
end

function [ret] = test_ref(basis, x0, y0)
    switch(basis)
        case 1
            ret = 2.0 * (x0 * x0 + y0 * y0) + 4.0 * x0 * y0 - 3.0 * (x0 + y0) + 1.0;
        case 2
            ret = x0 * (2.0 * x0 - 1.0);
        case 3
            ret = y0 * (2.0 * y0 - 1.0);
        case 4
            ret = 4.0 * x0 * (1.0 - x0 - y0);
        case 5
            ret = 4.0 * x0 * y0;
        case 6
            ret = 4.0 * y0 * (1.0 - x0 - y0);
        otherwise
            ret = 0.0;
    end
end

function [ret] = c(x, y)
    ret = 1.0;
end

function [ret] = f(x, y)
    ret = -y*(1-y)*(1-x-0.5*x*x)*exp(x+y)-x*(1-0.5*x)*(-3*y-y*y)*exp(x+y);
end

function [ret] = u(x, y)
    ret = x*y*(1-0.5*x)*(1-y)*exp(x+y);
end

function [ret] = grad_u(x, y)
    gx = y*(1-0.5*x)*(1-y)*exp(x+y) + x*y*(-0.5)*(1-y)*exp(x+y) + x*y*(1-0.5*x)*(1-y)*exp(x+y);
    gy = x*(1-0.5*x)*(1-y)*exp(x+y) + x*y*(1-0.5*x)*(-1)*exp(x+y) + x*y*(1-0.5*x)*(1-y)*exp(x+y);
    ret = [gx; gy];
end
