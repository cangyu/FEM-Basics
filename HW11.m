clear; close all; clc;

N_CASE = 6;
h = zeros(1, N_CASE);
err_velocity = zeros(3, N_CASE);
err_pressure = zeros(3, N_CASE);

for i = 1:N_CASE
    N = 2^i;
    h(i) = 1.0/N;
    
    xMin = 0.0;
    xMax = 1.0;
    yMin = -0.25;
    yMax = 0.0;
    
    N1 = round((xMax-xMin)/h(i));
    N2 = round((yMax-yMin)/h(i));
    h1 = (xMax-xMin)/N1;
    h2 = (yMax-yMin)/N2;
    
    t0 = 0.0;
    t1 = 1.0;
    
    theta = 0.5;
    dt0 = power(h(i), 1.5);
    loop_cnt = ceil((t1 - t0)/dt0);
    dt = (t1 - t0)/loop_cnt;
    
    fprintf("\nCASE%d: h=1/%d, hx=%g, hy=%g, dt=%g, theta=%g\n", i, N, h1, h2, dt, theta);
    
    [err_velocity(:, i), err_pressure(:, i)] = solve_2d_unsteady_stokes(xMin, xMax, yMin, yMax, N1, N2, t0, t1, loop_cnt, theta);
    
    fprintf("\n  Velocity:  |err|_inf=%e, |err|_L2=%e, |err|_H1=%e\n", err_velocity(1,i), err_velocity(2,i), err_velocity(3,i));
    fprintf("\n  Pressure:  |err|_inf=%e, |err|_L2=%e, |err|_H1=%e\n", err_pressure(1,i), err_pressure(2,i), err_pressure(3,i));
end

loglog(h, err_velocity(1,:), '-s')
hold on
loglog(h, err_velocity(2,:), '-s')
hold on
loglog(h, err_velocity(3,:), '-s')
grid on

loglog(h, err_pressure(1,:), '-+')
hold on
loglog(h, err_pressure(2,:), '-+')
hold on
loglog(h, err_pressure(3,:), '-+')
grid on

loglog([1e0, 1e-2], [1e1, 1e-1])
grid on
loglog([1e0, 1e-2], [1e0, 1e-4])
grid on
loglog([1e0, 1e-2], [1e-1, 1e-7])
grid on
legend('Velocity Inf', 'Velocity L2', 'Velocity H1-semi', 'Pressure Inf', 'Pressure L2', 'Pressure H1-semi', '1st-order slope', '2nd-order slope', '3rd-order slope', 'Location', 'southeast')

%% Coefficient assembly

function [errnorm_velocity, errnorm_pressure] = solve_2d_unsteady_stokes(x_min, x_max, y_min, y_max, N1, N2, t_start, t_end, n_iter, theta)
    global P T Pb Tb Jac
    global mu
    
    mu = 2.0;
    
    dt = (t_end - t_start) / n_iter;
    
    [P, T] = mesh_info_mat(x_min,x_max,y_min,y_max,N1,N2);
    [Pb, Tb] = fem_info_mat(x_min,x_max,y_min,y_max,N1,N2);
    [boundary_edge, boundary_node] = boundary_info_mat(N1, N2, T, Tb);
    
    Nlb_velocity = size(Tb, 1); % Num of local basis functions for velocity
    Nb_velocity = size(Pb, 2); % Num of global basis functions for velocity
    
    Nlb_pressure = size(T, 1); % Num of local basis functions for pressure
    Nb_pressure = size(P, 2); % Num of global basis functions for pressure
    
    N = size(T,2); % Num of mesh/FEM elements
    nbn = size(boundary_node, 2);
    nbe = size(boundary_edge, 2);
    
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
    
    % Assemble the stiffness matrix
    A1 = sparse(Nb_velocity, Nb_velocity);
    A2 = sparse(Nb_velocity, Nb_velocity);
    A3 = sparse(Nb_velocity, Nb_velocity);
    for n = 1:N
        for alpha = 1:Nlb_velocity % trial
            j = Tb(alpha, n);
            for beta = 1:Nlb_velocity % test
                i = Tb(beta, n);
                
                tmp = zeros(3, 1);
                for k = 1:gq_tri_n
                    x_next = gq_tri_x(n, k);
                    y = gq_tri_y(n, k);                    
                    
                    gpj = grad_velocity_trial(alpha, n, x_next, y);
                    gpi = grad_velocity_test(beta, n, x_next, y);
                    
                    tmp(1) = tmp(1) + gq_tri_w(k) * mu * gpj(1) * gpi(1);
                    tmp(2) = tmp(2) + gq_tri_w(k) * mu * gpj(2) * gpi(2);
                    tmp(3) = tmp(3) + gq_tri_w(k) * mu * gpj(1) * gpi(2);
                end
                tmp = tmp * abs(Jac(n));
                
                A1(i, j) = A1(i, j) + tmp(1);
                A2(i, j) = A2(i, j) + tmp(2);
                A3(i, j) = A3(i, j) + tmp(3);
            end
        end
    end
    A5 = sparse(Nb_velocity, Nb_pressure);
    A6 = sparse(Nb_velocity, Nb_pressure);
    for n = 1:N
        for alpha = 1:Nlb_pressure % trial
            j = T(alpha, n);
            for beta = 1:Nlb_velocity % test
                i = Tb(beta, n);
                
                tmp5 = 0.0;
                tmp6 = 0.0;
                for k = 1:gq_tri_n
                    x0 = gq_tri_x0(k);
                    y0 = gq_tri_y0(k);
                    x_next = gq_tri_x(n, k);
                    y = gq_tri_y(n, k);
                    
                    psij = pressure_trial_ref(alpha, x0, y0);
                    gphii = grad_velocity_test(beta, n, x_next, y);
                    
                    tmp5 = tmp5 + gq_tri_w(k) * -psij * gphii(1);
                    tmp6 = tmp6 + gq_tri_w(k) * -psij * gphii(2);
                end
                tmp5 = tmp5 * abs(Jac(n));
                tmp6 = tmp6 * abs(Jac(n));
                
                A5(i, j) = A5(i, j) + tmp5;
                A6(i, j) = A6(i, j) + tmp6;   
            end
        end
    end
    A01 = sparse(Nb_pressure, Nb_pressure);
    A = [2*A1+A2, A3, A5; A3.', 2*A2+A1, A6; A5.', A6.', A01];
    
    % Assemble the mass matrix
    Me = sparse(Nb_velocity, Nb_velocity);
    for n = 1:N
        for alpha = 1:Nlb_velocity % trial
            j = Tb(alpha, n);
            for beta = 1:Nlb_velocity % test
                i = Tb(beta, n);
                
                tmp = 0.0;
                for k = 1:gq_tri_n
                    x0 = gq_tri_x0(k);
                    y0 = gq_tri_y0(k);
                    
                    phii = velocity_test_ref(beta, x0, y0);
                    phij = velocity_trial_ref(alpha, x0, y0);
                    
                    tmp = tmp + gq_tri_w(k) * phij * phii;
                end
                tmp = tmp * abs(Jac(n));
                
                Me(i, j) = Me(i, j) + tmp;
            end
        end
    end
    A02 = sparse(Nb_velocity, Nb_pressure);
    A03 = sparse(Nb_velocity, Nb_velocity);
    M = [Me, A03, A02; A03, Me, A02; A02.', A02.', A01];
    
    M_tilde = M / (theta * dt);
    A_tilde = M_tilde + A;
    
    % Initialize
    u_sol = zeros(2, Nb_velocity);
    p_sol = zeros(1, Nb_pressure);
    for k = 1:Nb_velocity
        u_sol(:, k) = u(Pb(1, k), Pb(2, k), t_start);
    end
    for k = 1:Nb_pressure
        p_sol(k) = p(P(1, k), P(2, k), t_start);
    end
    x_cur = [u_sol(1, :).'; u_sol(2, :).'; p_sol.'];
    
    % Initial load vector
    b1 = zeros(Nb_velocity, 1);
    b2 = zeros(Nb_velocity, 1);
    bO = zeros(Nb_pressure, 1);
    for n = 1:N
        for beta = 1:Nlb_velocity % test
            i = Tb(beta, n);
            
            tmp = zeros(2, 1);
            for k = 1:gq_tri_n
                x0 = gq_tri_x0(k);
                y0 = gq_tri_y0(k);
                x_next = gq_tri_x(n, k);
                y = gq_tri_y(n, k);  
                
                fval = f(x_next, y, t_start);
                phii = velocity_test_ref(beta, x0, y0);
                
                tmp(1) = tmp(1) + gq_tri_w(k) * fval(1) * phii;
                tmp(2) = tmp(2) + gq_tri_w(k) * fval(2) * phii;
            end
            tmp = tmp * abs(Jac(n));
            
            b1(i) = b1(i) + tmp(1);
            b2(i) = b2(i) + tmp(2);
        end
    end
    b_cur = [b1; b2; bO];
    
    % Dirichlet Boundary for velocity
    for k = 1:nbn
        if boundary_node(1, k) == -1
            i = boundary_node(2, k);

            A_tilde(i, :) = 0;
            A_tilde(i, i) = 1;

            A_tilde(Nb_velocity + i, :) = 0;
            A_tilde(Nb_velocity + i, Nb_velocity + i) = 1;
        end
    end
    
    % Dirichlet Boundary for pressure
    node_flag = false(1, Nb_pressure);
    for k = 1:nbe
        if boundary_edge(1, k) == -1
            n_end1 = boundary_edge(3, k);
            n_end2 = boundary_edge(4, k);

            if(node_flag(n_end1) == false)
                node_flag(n_end1) = true;
                i = n_end1;

                A_tilde(2*Nb_velocity+i, :) = 0;
                A_tilde(2*Nb_velocity+i, 2*Nb_velocity+i) = 1;
            end

            if(node_flag(n_end2) == false)
                node_flag(n_end2) = true;
                i = n_end1;

                A_tilde(2*Nb_velocity+i, :) = 0;
                A_tilde(2*Nb_velocity+i, 2*Nb_velocity+i) = 1;
            end
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
        b1 = zeros(Nb_velocity, 1);
        b2 = zeros(Nb_velocity, 1);
        for n = 1:N
            for beta = 1:Nlb_velocity % test
                i = Tb(beta, n);

                tmp = zeros(2, 1);
                for k = 1:gq_tri_n
                    x0 = gq_tri_x0(k);
                    y0 = gq_tri_y0(k);
                    x_next = gq_tri_x(n, k);
                    y = gq_tri_y(n, k);  

                    fval = f(x_next, y, t_next);
                    phii = velocity_test_ref(beta, x0, y0);

                    tmp(1) = tmp(1) + gq_tri_w(k) * fval(1) * phii;
                    tmp(2) = tmp(2) + gq_tri_w(k) * fval(2) * phii;
                end
                tmp = tmp * abs(Jac(n));

                b1(i) = b1(i) + tmp(1);
                b2(i) = b2(i) + tmp(2);
            end
        end
        b_next = [b1; b2; bO];
        
        b_tilde = theta * b_next + (1.0-theta) * b_cur + M_tilde * x_cur;
        
        % Dirichlet Boundary for pressure
        node_flag = false(1, Nb_pressure);
        for k = 1:nbe
            if boundary_edge(1, k) == -1
                n_end1 = boundary_edge(3, k);
                n_end2 = boundary_edge(4, k);

                if(node_flag(n_end1) == false)
                    node_flag(n_end1) = true;
                    i = n_end1;

                    g = p(P(1, i), P(2, i));
                    A(2*Nb_velocity+i, :) = 0;
                    A(2*Nb_velocity+i, 2*Nb_velocity+i) = 1;
                    b(2*Nb_velocity+i) = g;
                end

                if(node_flag(n_end2) == false)
                    node_flag(n_end2) = true;
                    i = n_end1;

                    g = p(P(1, i), P(2, i));
                    A(2*Nb_velocity+i, :) = 0;
                    A(2*Nb_velocity+i, 2*Nb_velocity+i) = 1;
                    b(2*Nb_velocity+i) = g;
                end
            end
        end
                
        % Solve
        x_next = A_tilde\b_tilde;
        for i = 1:Nb_velocity
            u_sol(1, i) = x_next(i);
            u_sol(2, i) = x_next(Nb_velocity+i);
        end
        for i = 1:Nb_pressure
            p_sol(i) = x_next(2*Nb_velocity+i);
        end
        
        % Update
        t_cur = t_next;
        b_cur = b_next;
        x_cur = x_next;
    end
    
    % Check
    errnorm_velocity = zeros(1, 3); %inf, L2, semi-H1 respectively
    for n = 1:N
        for k = 1:gq_tri_n
            x0 = gq_tri_x0(k);
            y0 = gq_tri_y0(k);
            x_next = gq_tri_x(n, k);
            y = gq_tri_y(n, k);  
            
            w = zeros(2, 1);
            for i = 1:Nlb_velocity
                w = w + u_sol(:, Tb(i, n)) * velocity_trial_ref(i, x0, y0);
            end
            
            err = norm(w - u(x_next, y), Inf);
            if err > errnorm_velocity(1)
                errnorm_velocity(1) = err;
            end
        end
    end
    
    for n = 1:N
        res = 0.0;
        for k = 1:gq_tri_n
            x0 = gq_tri_x0(k);
            y0 = gq_tri_y0(k);
            x_next = gq_tri_x(n, k);
            y = gq_tri_y(n, k);  
            
            w = zeros(2, 1);
            for i = 1:Nlb_velocity
                w = w + u_sol(:, Tb(i, n)) * velocity_trial_ref(i, x0, y0);
            end
            
            err = norm(w - u(x_next, y))^2;
            res = res + gq_tri_w(k) * err;
        end
        res = res * abs(Jac(n));
        errnorm_velocity(2) = errnorm_velocity(2) + res;
    end
    errnorm_velocity(2) = sqrt(errnorm_velocity(2));
    
    for n = 1:N
        res = 0.0;
        for k = 1:gq_tri_n
            x_next = gq_tri_x(n, k);
            y = gq_tri_y(n, k);  
            
            w = zeros(2, 2);
            for i = 1:Nlb_velocity
                w = w + u_sol(:, Tb(i, n)) * grad_velocity_trial(i, n, x_next, y).';
            end
            
            err = norm(w - grad_u(x_next, y), 'fro')^2;
            res = res + gq_tri_w(k) * err;
        end
        res = res * abs(Jac(n));
        errnorm_velocity(3) = errnorm_velocity(3) + res;
    end
    errnorm_velocity(3) = sqrt(errnorm_velocity(3));
    
    errnorm_pressure = zeros(1, 3); %inf, L2, semi-H1 respectively
    for n = 1:N
        for k = 1:gq_tri_n
            x0 = gq_tri_x0(k);
            y0 = gq_tri_y0(k);
            x_next = gq_tri_x(n, k);
            y = gq_tri_y(n, k); 
                        
            w = 0.0;
            for i = 1:Nlb_pressure
                w = w + p_sol(T(i, n)) * pressure_trial_ref(i, x0, y0);
            end
            
            err = abs(w - p(x_next, y));
            if err > errnorm_pressure(1)
                errnorm_pressure(1) = err;
            end
        end
    end
    
    for n = 1:N
        res = 0.0;
        for k = 1:gq_tri_n
            x0 = gq_tri_x0(k);
            y0 = gq_tri_y0(k);
            x_next = gq_tri_x(n, k);
            y = gq_tri_y(n, k); 
                        
            w = 0.0;
            for i = 1:Nlb_pressure
                w = w + p_sol(T(i, n)) * pressure_trial_ref(i, x0, y0);
            end
            
            err = (w - p(x_next, y))^2;
            res = res + gq_tri_w(k) * err;
        end
        res = res * abs(Jac(n));
        errnorm_pressure(2) = errnorm_pressure(2) + res;
    end
    errnorm_pressure(2) = sqrt(errnorm_pressure(2));
    
    for n = 1:N
        res = 0.0;
        for k = 1:gq_tri_n
            x_next = gq_tri_x(n, k);
            y = gq_tri_y(n, k); 
                        
            w = zeros(2, 1);
            for i = 1:Nlb_pressure
                w = w + p_sol(T(i, n)) * grad_pressure_trial(i, n, x_next, y);
            end
            
            err = norm(w - grad_p(x_next, y))^2;
            res = res + gq_tri_w(k) * err;
        end
        res = res * abs(Jac(n));
        errnorm_pressure(3) = errnorm_pressure(3) + res;
    end
    errnorm_pressure(3) = sqrt(errnorm_pressure(3));
end

%% Mesh generation

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

%% Taylor-Hood Finite Element

function [ret] = grad_pressure_trial(basis, n, x, y)
    ret = grad_pressure_test(basis, n, x, y);
end

function [ret] = grad_pressure_test(basis, n, x, y)
    global P T Jac

    [x0, y0] = affine_mapping(n, x, y);
    gp = grad_pressure_test_ref(basis, x0, y0);
    
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

function [ret] = pressure_trial(basis, n, x, y)
    ret = pressure_test(basis, n, x, y);
end

function [ret] = pressure_test(basis, n, x, y)
    [x0, y0] = affine_mapping(n, x, y);
    ret = pressure_test_ref(basis, x0, y0);
end

function [ret] = grad_pressure_trial_ref(basis, x0, y0)
    ret = grad_pressure_test_ref(basis, x0, y0);
end

function [ret] = grad_pressure_test_ref(basis, x0, y0)
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

function [ret] = pressure_trial_ref(basis, x0, y0)
    ret = pressure_test_ref(basis, x0, y0);
end

function [ret] = pressure_test_ref(basis, x0, y0)
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

function [ret] = grad_velocity_trial(basis, n, x, y)
    ret = grad_velocity_test(basis, n, x, y);
end

function [ret] = grad_velocity_test(basis, n, x, y)
    global P T Jac

    [x0, y0] = affine_mapping(n, x, y);
    gp = grad_velocity_test_ref(basis, x0, y0);
    
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

function [ret] = velocity_trial(basis, n, x, y)
    ret = velocity_test(basis, n, x, y);
end

function [ret] = velocity_test(basis, n, x, y)
    [x0, y0] = affine_mapping(n, x, y);
    ret = velocity_test_ref(basis, x0, y0);
end

function [ret] = grad_velocity_trial_ref(basis, x0, y0)
    ret = grad_velocity_test_ref(basis, x0, y0);
end

function [ret] = grad_velocity_test_ref(basis, x0, y0)
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

function [ret] = velocity_trial_ref(basis, x0, y0)
    ret = velocity_test_ref(basis, x0, y0);
end

function [ret] = velocity_test_ref(basis, x0, y0)
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

%% Mapping

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

function [ret] = calc_elem_jacobi(P1, P2, P3)
    x1 = P1(1); y1 = P1(2);
    x2 = P2(1); y2 = P2(2);
    x3 = P3(1); y3 = P3(2);
    ret = (x2-x1)*(y3-y1)-(x3-x1)*(y2-y1);
end

%% Manufactured solution

function [ret] = f(x, y, t)
    global mu
    ret = zeros(2, 1);
    ret(1) = -2 * mu * (x^2 + y^2 + 0.5 * exp(-y)) * cos(2*pi*t) + pi^2 * cos(pi * x) * cos(2 * pi * y) * cos(2*pi*t) - 2 * pi * (x^2 * y^2 + exp(-y)) * sin(2*pi*t); 
    ret(2) = (mu * (4 * x * y - pi^3 * sin(pi * x)) + 2 * pi * (2 - pi * sin(pi * x)) * sin(2 * pi * y)) * cos(2*pi*t) - 2 * pi * sin(2*pi*t) * (-2.0/3 * x * y^3 + 2 - pi * sin(pi * x)); 
end

function [ret] = grad_u(x, y, t)
    ret = zeros(2, 2);
    ret(1, 1) = (2 * x * y^2)  * cos(2*pi*t);
    ret(1, 2) = (x^2 * 2 * y - exp(-y)) * cos(2*pi*t);
    ret(2, 1) = (-2.0/3 * y^3 - pi^2 * cos(pi * x)) * cos(2*pi*t);
    ret(2, 2) = (-2.0 * x * y^2) * cos(2*pi*t);
end

function [ret] = grad_p(x, y, t)
    ret = zeros(2, 1);
    ret(1) = - (-pi * cos(pi * x) * pi) * cos(2 * pi * y) * cos(2 * pi * t);
    ret(2) = -(2 - pi * sin(pi * x)) * (-sin(2 * pi * y) * 2 * pi) * cos(2 * pi * t);
end

function [ret] = u(x, y, t)
    ret = zeros(2, 1);
    ret(1) = (x^2 * y^2 + exp(-y)) * cos(2*pi*t);
    ret(2) = (-2.0/3 * x * y^3 + 2 - pi * sin(pi * x)) * cos(2*pi*t);
end

function [ret] = p(x, y, t)
    ret = -(2 - pi * sin(pi * x)) * cos(2 * pi * y) * cos(2 * pi * t);
end
