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
    
    fprintf("\nCASE%d: h=1/%d\n", i, N);
    
    [err_velocity(:, i), err_pressure(:, i)] = solve_2d_steady_navier_stokes(xMin, xMax, yMin, yMax, N1, N2);

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

function [errnorm_velocity, errnorm_pressure] = solve_2d_steady_navier_stokes(x_min, x_max, y_min, y_max, N1, N2)
    global P T Pb Tb Jac
    global mu
    
    mu = 2.0;
    
    [P, T] = mesh_info_mat(x_min,x_max,y_min,y_max,N1,N2);
    [Pb, Tb] = fem_info_mat(x_min,x_max,y_min,y_max,N1,N2);
    [boundary_edge, boundary_node] = boundary_info_mat(N1, N2, T, Tb);
    
    Nlb_velocity = size(Tb, 1); % Num of local basis functions for velocity
    Nb_velocity = size(Pb, 2); % Num of global basis functions for velocity
    
    Nlb_pressure = size(T, 1); % Num of local basis functions for pressure
    Nb_pressure = size(P, 2); % Num of global basis functions for pressure
    
    N = size(T, 2); % Num of mesh/FEM elements
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
                    x = gq_tri_x(n, k);
                    y = gq_tri_y(n, k);
                    
                    gpj = grad_velocity_trial(alpha, n, x, y);
                    gpi = grad_velocity_test(beta, n, x, y);
                    
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
                    x = gq_tri_x(n, k);
                    y = gq_tri_y(n, k);
                    
                    psij = pressure_trial_ref(alpha, x0, y0);
                    gphii = grad_velocity_test(beta, n, x, y);
                    
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
    AO1 = sparse(Nb_pressure, Nb_pressure);
    AO2 = sparse(Nb_velocity, Nb_pressure);
    A = [2*A1+A2, A3, A5; A3.', 2*A2+A1, A6; A5.', A6.', AO1];
    
    % Assemble the load vector
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
                x = gq_tri_x(n, k);
                y = gq_tri_y(n, k);
                
                fval = f(x, y);
                phii = velocity_test_ref(beta, x0, y0);
                
                tmp(1) = tmp(1) + gq_tri_w(k) * fval(1) * phii;
                tmp(2) = tmp(2) + gq_tri_w(k) * fval(2) * phii;
            end
            tmp = tmp * abs(Jac(n));
            
            b1(i) = b1(i) + tmp(1);
            b2(i) = b2(i) + tmp(2);
        end
    end
    b = [b1; b2; bO];
    
    u_sol = zeros(2, Nb_velocity);
    p_sol = zeros(1, Nb_pressure);
    
    % I.C. of u
    for k = 1:nbn
        if boundary_node(1, k) == -1
            i = boundary_node(2, k);
            u_sol(:, i) = u(Pb(1, i), Pb(2, i));
        end
    end
    
    converged=false;
    newton_iter = 0;
    while ~converged
        newton_iter=newton_iter+1;
        
        % gradient of velocity at previous iteration
        U = zeros(N, gq_tri_n, 2);
        grad_U = zeros(N, gq_tri_n, 2, 2);
        for n = 1:N
            for k = 1:gq_tri_n
                x0 = gq_tri_x0(k);
                y0 = gq_tri_y0(k);
                x = gq_tri_x(n, k);
                y = gq_tri_y(n, k);
                
                for alpha = 1:Nlb_velocity
                    j = Tb(alpha, n);
                    
                    phij = velocity_trial_ref(alpha, x0, y0);
                    gp = grad_velocity_trial(alpha, n, x, y);
                    
                    U(n, k, 1) = U(n, k, 1) + u_sol(1, j) * phij;
                    U(n, k, 2) = U(n, k, 2) + u_sol(2, j) * phij;

                    grad_U(n, k, 1, 1) = grad_U(n, k, 1, 1) + u_sol(1, j) * gp(1);
                    grad_U(n, k, 1, 2) = grad_U(n, k, 1, 2) + u_sol(1, j) * gp(2);
                    
                    grad_U(n, k, 2, 1) = grad_U(n, k, 2, 1) + u_sol(2, j) * gp(1);
                    grad_U(n, k, 2, 2) = grad_U(n, k, 2, 2) + u_sol(2, j) * gp(2);
                end
            end
        end
        
        % coefficients contributed by the convection term
        AN1 = sparse(Nb_velocity, Nb_velocity);
        AN2 = sparse(Nb_velocity, Nb_velocity);
        AN3 = sparse(Nb_velocity, Nb_velocity);
        AN4 = sparse(Nb_velocity, Nb_velocity);
        AN5 = sparse(Nb_velocity, Nb_velocity);
        AN6 = sparse(Nb_velocity, Nb_velocity);
        for n = 1:N
            for alpha = 1:Nlb_velocity % trial
                j = Tb(alpha, n);
                for beta = 1:Nlb_velocity % test
                    i = Tb(beta, n);

                    tmp = zeros(6, 1);
                    for k = 1:gq_tri_n
                        x0 = gq_tri_x0(k);
                        y0 = gq_tri_y0(k);
                        x = gq_tri_x(n, k);
                        y = gq_tri_y(n, k);
                        
                        phij = velocity_trial_ref(alpha, x0, y0);
                        phii = velocity_test_ref(beta, x0, y0);

                        gpj = grad_velocity_trial(alpha, n, x, y);

                        tmp(1) = tmp(1) + gq_tri_w(k) * grad_U(n, k, 1, 1) * phij * phii;
                        tmp(2) = tmp(2) + gq_tri_w(k) * U(n, k, 1) * gpj(1) * phii;
                        tmp(3) = tmp(3) + gq_tri_w(k) * U(n, k, 2) * gpj(2) * phii;
                        tmp(4) = tmp(4) + gq_tri_w(k) * grad_U(n, k, 1, 2) * phij * phii;
                        tmp(5) = tmp(5) + gq_tri_w(k) * grad_U(n, k, 2, 1) * phij * phii;
                        tmp(6) = tmp(6) + gq_tri_w(k) * grad_U(n, k, 2, 2) * phij * phii;
                    end
                    tmp = tmp * abs(Jac(n));

                    AN1(i, j) = AN1(i, j) + tmp(1);
                    AN2(i, j) = AN2(i, j) + tmp(2);
                    AN3(i, j) = AN3(i, j) + tmp(3);
                    AN4(i, j) = AN4(i, j) + tmp(4);
                    AN5(i, j) = AN5(i, j) + tmp(5);
                    AN6(i, j) = AN6(i, j) + tmp(6);
                end
            end
        end
        
        AN = [AN1 + AN2 + AN3, AN4, AO2; AN5, AN6 + AN2 + AN3, AO2; AO2.', AO2.', AO1];
        
        % source contributed by the convection term
        bN1 = zeros(Nb_velocity, 1);
        bN2 = zeros(Nb_velocity, 1);
        bN3 = zeros(Nb_velocity, 1);
        bN4 = zeros(Nb_velocity, 1);
        for n = 1:N
            for beta = 1:Nlb_velocity % test
                i = Tb(beta, n);
                
                tmp = zeros(4, 1);
                for k = 1:gq_tri_n
                    x0 = gq_tri_x0(k);
                    y0 = gq_tri_y0(k);

                    phii = velocity_test_ref(beta, x0, y0);

                    tmp(1) = tmp(1) + gq_tri_w(k) * U(n, k, 1) * grad_U(n, k, 1, 1) * phii;
                    tmp(2) = tmp(2) + gq_tri_w(k) * U(n, k, 2) * grad_U(n, k, 1, 2) * phii;
                    tmp(3) = tmp(3) + gq_tri_w(k) * U(n, k, 1) * grad_U(n, k, 2, 1) * phii;
                    tmp(4) = tmp(4) + gq_tri_w(k) * U(n, k, 2) * grad_U(n, k, 2, 2) * phii;
                end
                tmp = tmp * abs(Jac(n));

                bN1(i) = bN1(i) + tmp(1);
                bN2(i) = bN2(i) + tmp(2);
                bN3(i) = bN3(i) + tmp(3);
                bN4(i) = bN4(i) + tmp(4);
            end
        end
        bN = [bN1 + bN2; bN3 + bN4; bO];
        
        AN = A + AN;
        bN = b + bN;
        
        % Dirichlet Boundary for velocity
        for k = 1:nbn
            if boundary_node(1, k) == -1
                i = boundary_node(2, k);
                g = u(Pb(1, i), Pb(2, i));

                AN(i, :) = 0;
                AN(i, i) = 1;
                bN(i) = g(1);

                AN(Nb_velocity + i, :) = 0;
                AN(Nb_velocity + i, Nb_velocity + i) = 1;
                bN(Nb_velocity + i) = g(2);
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

                    g = p(P(1, i), P(2, i));
                    AN(2*Nb_velocity+i, :) = 0;
                    AN(2*Nb_velocity+i, 2*Nb_velocity+i) = 1;
                    bN(2*Nb_velocity+i) = g;
                end

                if(node_flag(n_end2) == false)
                    node_flag(n_end2) = true;
                    i = n_end2;

                    g = p(P(1, i), P(2, i));
                    AN(2*Nb_velocity+i, :) = 0;
                    AN(2*Nb_velocity+i, 2*Nb_velocity+i) = 1;
                    bN(2*Nb_velocity+i) = g;
                end
            end
        end

        % Solve
        xN = AN\bN;
        
        u_old = u_sol;
        p_old = p_sol;
        
        for i = 1:Nb_velocity
            u_sol(1, i) = xN(i);
            u_sol(2, i) = xN(Nb_velocity+i);
        end
        for i = 1:Nb_pressure
            p_sol(i) = xN(2*Nb_velocity+i);
        end
        
        newton_diff_velocity = 0.0;
        for i = 1:Nb_velocity
            newton_diff_velocity = newton_diff_velocity + norm(u_sol(:, i) - u_old(:, i), 'Inf');
        end
        newton_diff_velocity = newton_diff_velocity / Nb_velocity;
        newton_diff_pressure = 0.0;
        for i = 1:Nb_pressure
            newton_diff_pressure = newton_diff_pressure + abs(p_sol(i) - p_old(i));
        end
        newton_diff_pressure = newton_diff_pressure / Nb_pressure;
        
        fprintf("  Iter%d: diff_V=%g, diff_p=%g\n", newton_iter, newton_diff_velocity, newton_diff_pressure);
        if newton_diff_velocity < 1e-8 && newton_diff_pressure < 1e-8
            converged = true;

            % Check
            errnorm_velocity = zeros(1, 3); %inf, L2, semi-H1 respectively
            for n = 1:N
                for k = 1:gq_tri_n
                    x0 = gq_tri_x0(k);
                    y0 = gq_tri_y0(k);
                    x = gq_tri_x(n, k);
                    y = gq_tri_y(n, k);  

                    w = zeros(2, 1);
                    for i = 1:Nlb_velocity
                        w = w + u_sol(:, Tb(i, n)) * velocity_trial_ref(i, x0, y0);
                    end

                    err = norm(w - u(x, y), Inf);
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
                    x = gq_tri_x(n, k);
                    y = gq_tri_y(n, k);  

                    w = zeros(2, 1);
                    for i = 1:Nlb_velocity
                        w = w + u_sol(:, Tb(i, n)) * velocity_trial_ref(i, x0, y0);
                    end

                    err = norm(w - u(x, y))^2;
                    res = res + gq_tri_w(k) * err;
                end
                res = res * abs(Jac(n));
                errnorm_velocity(2) = errnorm_velocity(2) + res;
            end
            errnorm_velocity(2) = sqrt(errnorm_velocity(2));

            for n = 1:N
                res = 0.0;
                for k = 1:gq_tri_n
                    x = gq_tri_x(n, k);
                    y = gq_tri_y(n, k);  

                    w = zeros(2, 2);
                    for i = 1:Nlb_velocity
                        w = w + u_sol(:, Tb(i, n)) * grad_velocity_trial(i, n, x, y).';
                    end

                    err = norm(w - grad_u(x, y), 'fro')^2;
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
                    x = gq_tri_x(n, k);
                    y = gq_tri_y(n, k); 

                    w = 0.0;
                    for i = 1:Nlb_pressure
                        w = w + p_sol(T(i, n)) * pressure_trial_ref(i, x0, y0);
                    end

                    err = abs(w - p(x, y));
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
                    x = gq_tri_x(n, k);
                    y = gq_tri_y(n, k); 

                    w = 0.0;
                    for i = 1:Nlb_pressure
                        w = w + p_sol(T(i, n)) * pressure_trial_ref(i, x0, y0);
                    end

                    err = (w - p(x, y))^2;
                    res = res + gq_tri_w(k) * err;
                end
                res = res * abs(Jac(n));
                errnorm_pressure(2) = errnorm_pressure(2) + res;
            end
            errnorm_pressure(2) = sqrt(errnorm_pressure(2));

            for n = 1:N
                res = 0.0;
                for k = 1:gq_tri_n
                    x = gq_tri_x(n, k);
                    y = gq_tri_y(n, k); 

                    w = zeros(2, 1);
                    for i = 1:Nlb_pressure
                        w = w + p_sol(T(i, n)) * grad_pressure_trial(i, n, x, y);
                    end

                    err = norm(w - grad_p(x, y))^2;
                    res = res + gq_tri_w(k) * err;
                end
                res = res * abs(Jac(n));
                errnorm_pressure(3) = errnorm_pressure(3) + res;
            end
            errnorm_pressure(3) = sqrt(errnorm_pressure(3));
        end
    end
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

function [ret] = f(x, y)
    global mu
    ret = zeros(2, 1);
    ret(1) = -2 * mu * (x^2 + y^2 + 0.5 * exp(-y)) + pi^2 * cos(pi * x) * cos(2 * pi * y) + 2 * x * y^2 * (x^2 * y^2 + exp(-y)) + (-2/3 * x * y^3 + 2 - pi * sin(pi * x))*(2 * x^2 * y - exp(-y)); 
    ret(2) = mu * (4 * x * y - pi^3 * sin(pi * x)) + 2 * pi * (2 - pi * sin(pi * x)) * sin(2 * pi * y) + (x^2 * y^2 + exp(-y)) * (-2/3*y^3 - pi^2 * cos(pi * x)) - 2*x*y^2 * (-2/3*x*y^3 + 2 - pi * sin(pi * x)); 
end

function [ret] = grad_u(x, y)
    ret = zeros(2, 2);
    ret(1, 1) = 2 * x * y^2;
    ret(1, 2) = x^2 * 2 * y - exp(-y);
    ret(2, 1) = -2.0/3 * y^3 - pi^2 * cos(pi * x);
    ret(2, 2) = -2.0 * x * y^2;
end

function [ret] = grad_p(x, y)
    ret = zeros(2, 1);
    ret(1) = - (-pi * cos(pi * x) * pi) * cos(2 * pi * y);
    ret(2) = -(2 - pi * sin(pi * x)) * (-sin(2 * pi * y) * 2 * pi);
end

function [ret] = u(x, y)
    ret = zeros(2, 1);
    ret(1) = x^2 * y^2 + exp(-y);
    ret(2) = -2.0/3 * x * y^3 + 2 - pi * sin(pi * x);
end

function [ret] = p(x, y)
    ret = -(2 - pi * sin(pi * x)) * cos(2 * pi * y);
end
