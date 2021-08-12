clear; close all; clc;

N_CASE = 6;
h = zeros(1, N_CASE);
err = zeros(3, N_CASE);

for i = 1:N_CASE
    N = 2^i;
    theta = 0.5;
    fprintf("\nCASE%d: theta=%g, ", i, theta);
    [ch1, ch2, ce] = solve_2d_parabolic_pde(N, N, theta, 0.0, 1.0);
    h(i) = sqrt(ch1*ch2);
    err(1,i) = ce(1);
    err(2,i) = ce(2);
    err(3,i) = ce(3);
    fprintf("\n  |err|_inf=%e, |err|_L2=%e, |err|_H1=%e\n", err(1,i), err(2,i), err(3,i));
end

loglog(h, err(1,:), '-s')
hold on
loglog(h, err(2,:), '-s')
hold on
loglog(h, err(3,:), '-s')
grid on
legend('inf','L2','semi-H1','Location','northwest')

function [h1, h2, errnorm] = solve_2d_parabolic_pde(N1, N2, theta, t_start, t_end)
    global P T Jac

    x_min = 0.0; x_max = 2.0; h1 = (x_max-x_min)/N1;
    y_min = 0.0; y_max = 1.0; h2 = (y_max-y_min)/N2;
    
    dt = sqrt(h1 * h2);
    
    fprintf("h1=%g, h2=%g, dt=%g\n", h1, h2, dt);
        
    [P, T] = mesh_info_mat(x_min,x_max,y_min,y_max,N1,N2);
    [boundary_edge, boundary_node] = boundary_info_mat(N1, N2, T);
    
    Nlb = 3;
    N = 2*N1*N2;
    Nb = size(P, 2);
    nbn = size(boundary_node, 2);
    nbe = size(boundary_edge, 2);
    
    % Element jacobian
    Jac = zeros(N, 1);
    for i = 1:N1
        for j = 1:N2
            quad_idx = (i-1)*N2 + j;
            tri0_idx = 2*quad_idx - 1;
            tri1_idx = tri0_idx + 1;
            
            Jac(tri0_idx) = calc_elem_jacobi(P(:, T(1, tri0_idx)), P(:, T(2, tri0_idx)), P(:, T(3, tri0_idx)));
            Jac(tri1_idx) = calc_elem_jacobi(P(:, T(1, tri1_idx)), P(:, T(2, tri1_idx)), P(:, T(3, tri1_idx)));
        end
    end
    
    % Gauss quadrature coefficients
    gq_tri_x0 = [1.0/3, 1.0/5, 3.0/5, 1.0/5];
    gq_tri_y0 = [1.0/3, 1.0/5, 1.0/5, 3.0/5];
    gq_tri_w = [-27.0/96, 25.0/96, 25.0/96, 25.0/96];
    gq_tri_n = 4;
    
    gq_lin_s0 = [-sqrt(3/5),0,sqrt(3/5)];
    gq_lin_w = [5/9, 8/9, 5/9];
    gq_lin_n = 3;
    
    % Assemble the mass matrix
    M = zeros(Nb, Nb);
    for n = 1:N
        for alpha = 1:Nlb % trial
            for beta = 1:Nlb % test
                i = T(beta, n);
                j = T(alpha, n);
                
                tmp = 0.0;
                for k = 1:gq_tri_n
                    x0 = gq_tri_x0(k); 
                    y0 = gq_tri_y0(k);
                    tmp = tmp + gq_tri_w(k) * trial_ref(alpha, x0, y0) * test_ref(beta, x0, y0);
                end
                tmp = tmp * abs(Jac(n));
                
                M(i, j) = M(i, j) + tmp;
            end
        end
    end
    
    % Assemble the stiffness matrix
    A = zeros(Nb, Nb);
    for n = 1:N
        for alpha = 1:Nlb
            for beta = 1:Nlb
                i = T(beta, n);
                j = T(alpha, n);
                
                tmp = 0.0;
                for k = 1:gq_tri_n
                    [x, y] = affine_mapping_back(n, gq_tri_x0(k), gq_tri_y0(k));
                    tmp1 = grad_trial(alpha, n, x, y);
                    tmp2 = grad_test(beta, n, x, y);
                    tmp = tmp + gq_tri_w(k) * c(x, y) * dot(tmp1, tmp2);
                end
                tmp = tmp * abs(Jac(n));
                
                A(i, j) = A(i, j) + tmp;
            end
        end
    end
    
    % In this case, the stiffness matrix does NOT change with time.
    A_tilde = M / dt + theta * A;
    A_res = M / dt - (1.0-theta) * A;
    % Dirichlet Boundary
    for k = 1:nbn
        if boundary_node(1, k) == -1
            i = boundary_node(2, k);
            A_tilde(i, :) = 0;
            A_tilde(i, i) = 1;
        end
    end
    
    % Initialize
    u_sol = zeros(Nb, 1);
    for k = 1:Nb
        u_sol(k) = u(P(1, k), P(2, k), t_start);
    end
    b_cur = zeros(Nb, 1);
    for n = 1:N
        for beta = 1:Nlb
            i = T(beta, n);

            tmp = 0.0;
            for k = 1:gq_tri_n
                [x, y] = affine_mapping_back(n, gq_tri_x0(k), gq_tri_y0(k));
                tmp = tmp + gq_tri_w(k) * f(x, y, t_start) * test(beta, n, x, y);
            end
            tmp = tmp * abs(Jac(n));

            b_cur(i) = b_cur(i) + tmp;
        end
    end
    
    % Time-Marching
    t_cur = t_start;
    cnt = 0;
    while t_cur < t_end
        % To arrive at prescribed ending time exactly.
        t_remain = t_end - t_cur;
        if t_remain < dt
            dt = t_remain;
            A_tilde = M/dt + theta*A;
            A_res = M/dt - (1.0-theta)*A;
            % Dirichlet Boundary
            for k = 1:nbn
                if boundary_node(1, k) == -1
                    i = boundary_node(2, k);
                    A_tilde(i, :) = 0;
                    A_tilde(i, i) = 1;
                end
            end
        end
        t_next = t_cur + dt;
        cnt = cnt + 1;
        fprintf("  iter%4d: t_cur=%10g, t_next=%10g\n", cnt, t_cur, t_next);

        % Assemble the load vector
        b_next = zeros(Nb, 1);
        for n = 1:N
            for beta = 1:Nlb
                i = T(beta, n);

                tmp = 0.0;
                for k = 1:gq_tri_n
                    [x, y] = affine_mapping_back(n, gq_tri_x0(k), gq_tri_y0(k));
                    tmp = tmp + gq_tri_w(k) * f(x, y, t_next) * test(beta, n, x, y);
                end
                tmp = tmp * abs(Jac(n));

                b_next(i) = b_next(i) + tmp;
            end
        end
        
        b_tilde = theta * b_next + (1.0-theta) * b_cur + A_res*u_sol; 
        
        % Dirichlet Boundary
        for k = 1:nbn
            if boundary_node(1, k) == -1
                i = boundary_node(2, k);
                b_tilde(i) = u(P(1, i), P(2, i), t_next);
            end
        end
        
        % Solve
        u_sol = A_tilde\b_tilde;
        
        % Update
        t_cur = t_next;
        b_cur = b_next;
    end

    % Check
    errnorm = zeros(1, 3); %inf, L2, semi-H1 respectively
    for n = 1:N
        for k = 1:gq_tri_n
            x0 = gq_tri_x0(k);
            y0 = gq_tri_y0(k);
            
            [x, y] = affine_mapping_back(n, x0, y0);
            
            w = 0.0;
            for i = 1:Nlb
                w = w + u_sol(T(i, n)) * trial_ref(i, x0, y0);
            end
            
            err = abs(w - u(x, y, t_end));
            if err > errnorm(1)
                errnorm(1) = err;
            end
        end
    end
    
    for n = 1:N
        res = 0.0;
        for k = 1:gq_tri_n
            x0 = gq_tri_x0(k);
            y0 = gq_tri_y0(k);
            
            [x, y] = affine_mapping_back(n, x0, y0);
            
            w = 0.0;
            for i = 1:Nlb
                w = w + u_sol(T(i, n)) * trial_ref(i, x0, y0);
            end
            
            err = (w - u(x, y, t_end))^2;
            res = res + gq_tri_w(k) * err;
        end
        res = res * abs(Jac(n));
        errnorm(2) = errnorm(2) + res;
    end
    errnorm(2) = sqrt(errnorm(2));
    
    for n = 1:N
        res = 0.0;
        for k = 1:gq_tri_n
            x0 = gq_tri_x0(k);
            y0 = gq_tri_y0(k);
            
            [x, y] = affine_mapping_back(n, x0, y0);
            
            w = zeros(2, 1);
            for i = 1:Nlb
                w = w + u_sol(T(i, n)) * grad_trial(i, n, x, y);
            end
            
            err = norm(w - grad_u(x, y, t_end))^2;
            res = res + gq_tri_w(k) * err;
        end
        res = res * abs(Jac(n));
        errnorm(3) = errnorm(3) + res;
    end
    errnorm(3) = sqrt(errnorm(3));
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
            ret = [-1; -1];
        case 2
            ret = [1; 0];
        case 3
            ret = [0; 1];
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
            ret = 1.0 - x0 - y0;
        case 2
            ret = x0;
        case 3
            ret = y0;
        otherwise
            ret = 0;
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

function [bdry_edge, bdry_node] = boundary_info_mat(n1, n2, T)
    bdry_edge = zeros(4, 2*(n1+n2));
    bdry_node = zeros(2, 2*(n1+n2));
    
    % Bottom
    for k = 1:n1
        edge_idx = k;
        elem_idx = 1 + (k-1)*n2*2;
        node_idx = edge_idx;

        bdry_edge(1, edge_idx) = -1;
        bdry_edge(2, edge_idx) = elem_idx;
        bdry_edge(3, edge_idx) = T(1, elem_idx);
        bdry_edge(4, edge_idx) = T(2, elem_idx);
        
        bdry_node(1, node_idx) = -1;
        bdry_node(2, node_idx) = T(1, elem_idx);
    end
    
    % Right
    for k = 1:n2
        edge_idx = k+n1;
        elem_idx = 2*n2*(n1-1) + 2*k;
        node_idx = edge_idx;

        bdry_edge(1, edge_idx) = -1;
        bdry_edge(2, edge_idx) = elem_idx;
        bdry_edge(3, edge_idx) = T(2, elem_idx);
        bdry_edge(4, edge_idx) = T(3, elem_idx);
        
        bdry_node(1, node_idx) = -1;
        bdry_node(2, node_idx) = T(2, elem_idx);
    end
    
    % Top
    for k = 1:n1
        edge_idx = k+n2+n1;
        elem_idx = 2*n1*n2 - 2*n2*(k-1);
        node_idx = edge_idx;
        
        bdry_edge(1, edge_idx) = -1;
        bdry_edge(2, edge_idx) = elem_idx;
        bdry_edge(3, edge_idx) = T(3, elem_idx);
        bdry_edge(4, edge_idx) = T(1, elem_idx);
        
        bdry_node(1, node_idx) = -1;
        bdry_node(2, node_idx) = T(3, elem_idx);
    end
    
    % Left
    for k = 1:n2
        edge_idx = k+2*n1+n2;
        elem_idx = 2*n2 - (2*k-1);
        node_idx = edge_idx;

        bdry_edge(1, edge_idx) = -1;
        bdry_edge(2, edge_idx) = elem_idx;
        bdry_edge(3, edge_idx) = T(3, elem_idx);
        bdry_edge(4, edge_idx) = T(1, elem_idx);
        
        bdry_node(1, node_idx) = -1;
        bdry_node(2, node_idx) = T(3, elem_idx);
    end
end

function [ret] = c(x, y, t)
    ret = 2.0;
end

function [ret] = f(x, y, t)
    ret = -3.0*exp(x+y+t);
end

function [ret] = u(x, y, t)
    ret = exp(x+y+t);
end

function [ret] = grad_u(x, y, t)
    gx = exp(x+y+t);
    gy = exp(x+y+t);
    ret = [gx; gy];
end
