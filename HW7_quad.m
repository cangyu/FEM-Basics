clear; close all; clc;

N_CASE = 5;
h = zeros(1, N_CASE);
err = zeros(3, N_CASE);

for i = 1:N_CASE
    N = 2^i;
    h(i) = 1.0/N;
    
    xMin = 0.0; 
    xMax = 2.0; 
    yMin = 0.0; 
    yMax = 1.0; 
    
    N1 = round((xMax-xMin)/h(i));
    N2 = round((yMax-yMin)/h(i));
    h1 = (xMax-xMin)/N1;
    h2 = (yMax-yMin)/N2;
    
    t0 = 0.0;
    t1 = 1.0;
    
    theta = 1.0;
    dt0 = 8.0*power(h(i), 3.0);
    loop_cnt = ceil((t1 - t0)/dt0);
    dt = (t1 - t0)/loop_cnt;    
    
    fprintf("\nCASE%d: h=1/%d, hx=%g, hy=%g, dt=%g, theta=%g\n", i, N, h1, h2, dt, theta);
    
    err(:, i) = solve_2d_parabolic_pde(xMin, xMax, yMin, yMax, N1, N2, t0, t1, loop_cnt, theta);

    fprintf("\n  |err|_inf=%e, |err|_L2=%e, |err|_H1=%e\n", err(1,i), err(2,i), err(3,i));
end

loglog(h, err(1,:), '-s')
hold on
loglog(h, err(2,:), '-s')
hold on
loglog(h, err(3,:), '-s')
grid on
% legend('inf', 'L2', 'semi-H1', 'Location', 'southeast')
loglog([1e0, 1e-1], [1e1, 1e0])
grid on
loglog([1e0, 1e-1], [1e1, 1e-1])
grid on
loglog([1e0, 1e-1], [1e0, 1e-3])
grid on
legend('inf', 'L2', 'semi-H1', '1st-order', '2nd-order', '3rd-order', 'Location', 'southeast')

function [errnorm] = solve_2d_parabolic_pde(x_min, x_max, y_min, y_max, N1, N2, t_start, t_end, n_iter, theta)
    global P T Pb Tb Jac
    
    dt = (t_end - t_start) / n_iter;
    
    [P, T] = mesh_info_mat(x_min,x_max,y_min,y_max,N1,N2);
    [Pb, Tb] = fem_info_mat(x_min,x_max,y_min,y_max,N1,N2);
    [boundary_edge, boundary_node] = boundary_info_mat(N1, N2, T, Tb);
    
    Nlb = size(Tb, 1); % Num of local basis functions
    Nb = size(Pb, 2); % Num of global basis functions(fem unknowns)
    N = size(T,2); % Num of mesh/fem elements
    Nm = size(P, 2); % Num of mesh nodes
    nbn = size(boundary_node, 2); % Num of boundary nodes
    nbe = size(boundary_edge, 2); % Num of boundary edges
    
    % Element jacobian
    Jac = zeros(N, 1);
    for i = 1:N
        p1 = P(:, T(1, i));
        p2 = P(:, T(2, i));
        p3 = P(:, T(3, i));
        Jac(i) = calc_elem_jacobi(p1, p2, p3);
    end
    
    % Gauss quadrature coordinates & coefficients
    gq_tri_n = 4;
    gq_tri_x0 = [1.0/3, 1.0/5, 3.0/5, 1.0/5];
    gq_tri_y0 = [1.0/3, 1.0/5, 1.0/5, 3.0/5];
    gq_tri_w = [-27.0/96, 25.0/96, 25.0/96, 25.0/96];
    gq_tri_x = zeros(N, gq_tri_n);
    gq_tri_y = zeros(N, gq_tri_n);
    for n = 1:N
        for k = 1:gq_tri_n
            x0 = gq_tri_x0(k); 
            y0 = gq_tri_y0(k);
            [gq_tri_x(n, k), gq_tri_y(n, k)] = affine_mapping_back(n, x0, y0);
        end
    end

    % Assemble the mass matrix
    M = sparse(Nb, Nb);
    for n = 1:N
        for alpha = 1:Nlb % trial
            j = Tb(alpha, n);
            for beta = 1:Nlb % test
                i = Tb(beta, n);
                
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
    A = sparse(Nb, Nb);
    for n = 1:N
        for alpha = 1:Nlb % trial
            j = Tb(alpha, n);
            for beta = 1:Nlb % test
                i = Tb(beta, n);
                
                tmp = 0.0;
                for k = 1:gq_tri_n
                    x = gq_tri_x(n, k);
                    y = gq_tri_y(n, k);                    
                    tmp1 = grad_trial(alpha, n, x, y);
                    tmp2 = grad_test(beta, n, x, y);
                    tmp3 = c(x, y, t_start) * dot(tmp1, tmp2);
                    tmp = tmp + gq_tri_w(k) * tmp3;
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
        u_sol(k) = u(Pb(1, k), Pb(2, k), t_start);
    end
    
    b_cur = zeros(Nb, 1);
    for n = 1:N
        for beta = 1:Nlb
            i = Tb(beta, n);

            tmp = 0.0;
            for k = 1:gq_tri_n
                x0 = gq_tri_x0(k);
                y0 = gq_tri_y0(k);
                x = gq_tri_x(n, k);
                y = gq_tri_y(n, k);  
                tmp = tmp + gq_tri_w(k) * f(x, y, t_start) * test_ref(beta, x0, y0);
            end
            tmp = tmp * abs(Jac(n));

            b_cur(i) = b_cur(i) + tmp;
        end
    end
    
    % Time-Marching
    t_cur = t_start;
    cnt = 0;
    while cnt < n_iter
        t_next = t_cur + dt;
        cnt = cnt + 1;
        fprintf("  iter%4d: t_cur=%10g, t_next=%10g\n", cnt, t_cur, t_next);

        % Assemble the load vector
        b_next = zeros(Nb, 1);
        for n = 1:N
            for beta = 1:Nlb
                i = Tb(beta, n);

                tmp = 0.0;
                for k = 1:gq_tri_n
                    x0 = gq_tri_x0(k);
                    y0 = gq_tri_y0(k);
                    x = gq_tri_x(n, k);
                    y = gq_tri_y(n, k);  
                    tmp = tmp + gq_tri_w(k) * f(x, y, t_next) * test_ref(beta, x0, y0);
                end
                tmp = tmp * abs(Jac(n));

                b_next(i) = b_next(i) + tmp;
            end
        end
        
        b_tilde = theta * b_next + (1.0-theta) * b_cur + A_res * u_sol; 
        
        % Dirichlet Boundary
        for k = 1:nbn
            if boundary_node(1, k) == -1
                i = boundary_node(2, k);
                b_tilde(i) = u(Pb(1, i), Pb(2, i), t_next);
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
            x = gq_tri_x(n, k);
            y = gq_tri_y(n, k);  
            
            w = 0.0;
            for i = 1:Nlb
                w = w + u_sol(Tb(i, n)) * trial_ref(i, x0, y0);
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
            x = gq_tri_x(n, k);
            y = gq_tri_y(n, k);  
            
            w = 0.0;
            for i = 1:Nlb
                w = w + u_sol(Tb(i, n)) * trial_ref(i, x0, y0);
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
            x = gq_tri_x(n, k);
            y = gq_tri_y(n, k);  
            
            w = zeros(2, 1);
            for i = 1:Nlb
                w = w + u_sol(Tb(i, n)) * grad_trial(i, n, x, y);
            end
            
            err = norm(w - grad_u(x, y, t_end))^2;
            res = res + gq_tri_w(k) * err;
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
