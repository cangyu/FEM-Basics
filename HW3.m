clear; close all; clc;

N_CASE = 9;
h = zeros(1, N_CASE);
err = zeros(3, N_CASE);

for i = 1:N_CASE
    N = 2^i;
    [ch, ce] = solve_1d_elliptic_pde(N);
    h(i) = ch;
    err(:,i) = ce;
    fprintf("h=1/%d, |err|_inf=%e, |err|_L2=%e, |err|_H1=%e\n", N, ce(1), ce(2), ce(3));
end

loglog(h, err(1,:), '-s')
hold on
loglog(h, err(2,:), '-s')
hold on
loglog(h, err(3,:), '-s')
grid on
legend('inf','L2','H1','Location','northwest')

function [h, errnorm] = solve_1d_elliptic_pde(N)
    xa = 0.0;
    xb = 1.0;

    h = (xb-xa)/N;
    Nm = N+1;

    P = zeros(1, Nm);
    for j = 1:Nm
        P(j) = xa + (j-1)*h;
    end

    T = zeros(2, N);
    for j = 1:N
        T(1, j) = j;
        T(2, j) = j+1;
    end

    yh = h/2;
    Nb = 2*N+1;

    Pb = zeros(1, Nb);
    for j = 1:Nb
        Pb(j) = xa + (j-1) * yh;
    end

    Tb = zeros(3, N);
    for j = 1:N
        Tb(1, j) = 2*j-1;
        Tb(2, j) = 2*j+1;
        Tb(3, j) = 2*j;
    end

    Nlb_trial = 3;
    Nlb_test = 3;

    A = zeros(Nb, Nb);
    b = zeros(Nb, 1);

    gauss_quad_s=[-sqrt(3/5),0,sqrt(3/5)];
    gauss_quad_w=[5/9, 8/9, 5/9];
    gauss_quad_n=3;

    for n = 1:N
        xn0 = P(n); xn1 = P(n+1);
        xnh = xn1-xn0; xnh2 = xnh*xnh;

        % gauss quadrature points
        gauss_quad_x = zeros(1,gauss_quad_n);
        gauss_quad_x0 = zeros(1,gauss_quad_n);
        for i = 1:gauss_quad_n
            gauss_quad_x(i) = ((xn1+xn0)+(xn1-xn0)*gauss_quad_s(i))/2;
            gauss_quad_x0(i) = (gauss_quad_x(i)-xn0)/(xn1-xn0);
        end

        % coefficient matrix
        for alpha = 1:Nlb_trial
            for beta = 1:Nlb_test
                % quadrature
                r = 0;
                for i = 1:gauss_quad_n
                    xsp = gauss_quad_x(i);
                    xsp0 = gauss_quad_x0(i);
                    vsp = c(xsp)*trial_prime(xsp0, alpha)*test_prime(xsp0, beta)/xnh2;
                    r = r + gauss_quad_w(i) * vsp;
                end
                r = r * (xn1-xn0)/2;

                % assemble
                gi = Tb(beta, n); gj = Tb(alpha, n);
                A(gi, gj) = A(gi, gj) + r;
            end
        end

        % load vector
        for beta = 1:Nlb_test
            % quadrature
            r = 0;
            for i = 1:gauss_quad_n
                xsp = gauss_quad_x(i);
                xsp0 = gauss_quad_x0(i);
                vsp = f(xsp) * test(xsp0, beta);
                r = r + gauss_quad_w(i) * vsp;
            end
            r = r * (xn1-xn0)/2;

            % assemble
            gi = Tb(beta, n);
            b(gi) = b(gi) + r;
        end
    end

    % boundary condition
    qa = 1.0;
    pa = 1.0;
    A(1,1) = A(1,1) + c(xa)*qa;
    b(1) = b(1) - c(xa)*pa;

    A(Nb,:) = 0;
    A(Nb,Nb) = 1;
    b(Nb) = u(xb);

    % solve
    sol = A \ b;

    % check
    errnorm = zeros(1, 3); %inf, L2, H1 respectively

    res = zeros(N, 1);
    for n = 1:N
        xn0 = P(n); xn1 = P(n+1);
        
        % gauss quadrature points
        gauss_quad_x = zeros(1,gauss_quad_n);
        gauss_quad_x0 = zeros(1,gauss_quad_n);
        for i = 1:gauss_quad_n
            gauss_quad_x(i) = ((xn1+xn0)+(xn1-xn0)*gauss_quad_s(i))/2;
            gauss_quad_x0(i) = (gauss_quad_x(i)-xn0)/(xn1-xn0);
        end
        
        % error on each element
        w = zeros(1, gauss_quad_n);
        e = zeros(1, gauss_quad_n);
        for i = 1:gauss_quad_n
            w(i) = 0.0;
            for k = 1:Nlb_trial
                w(i) = w(i) + sol(Tb(k, n))*trial(gauss_quad_x0(i), k);
            end
            e(i) = abs(u(gauss_quad_x(i))-w(i)); % inf norm
        end
        
        res(n) = max(e);
    end
    errnorm(1) = max(res);
    
    res = 0.0;
    for n = 1:N
        xn0 = P(n); xn1 = P(n+1);
        
        % gauss quadrature points
        gauss_quad_x = zeros(1,gauss_quad_n);
        gauss_quad_x0 = zeros(1,gauss_quad_n);
        for i = 1:gauss_quad_n
            gauss_quad_x(i) = ((xn1+xn0)+(xn1-xn0)*gauss_quad_s(i))/2;
            gauss_quad_x0(i) = (gauss_quad_x(i)-xn0)/(xn1-xn0);
        end
        
        % error on each element
        w = zeros(1, gauss_quad_n);
        e = 0.0;
        for i = 1:gauss_quad_n
            w(i) = 0.0;
            for k = 1:Nlb_trial
                w(i) = w(i) + sol(Tb(k, n))*trial(gauss_quad_x0(i), k);
            end
            e = e + gauss_quad_w(i) * power(u(gauss_quad_x(i))-w(i), 2); % L2 norm
        end
        e = e * (xn1-xn0)/2;
        
        res = res + e;
    end
    errnorm(2) = sqrt(res);
    
    res = 0.0;
    for n = 1:N
        xn0 = P(n); xn1 = P(n+1);
        xnh = xn1-xn0;
        
        % gauss quadrature points
        gauss_quad_x = zeros(1,gauss_quad_n);
        gauss_quad_x0 = zeros(1,gauss_quad_n);
        for i = 1:gauss_quad_n
            gauss_quad_x(i) = ((xn1+xn0)+(xn1-xn0)*gauss_quad_s(i))/2;
            gauss_quad_x0(i) = (gauss_quad_x(i)-xn0)/(xn1-xn0);
        end
        
        % error on each element
        w = zeros(1, gauss_quad_n);
        e = 0.0;
        for i = 1:gauss_quad_n
            w(i) = 0.0;
            for k = 1:Nlb_trial
                w(i) = w(i) + sol(Tb(k, n))*trial_prime(gauss_quad_x0(i), k)/xnh;
            end
            e = e + gauss_quad_w(i) * power(u_prime(gauss_quad_x(i))-w(i), 2); % H1 semi-norm
        end
        e = e * (xn1-xn0)/2;
        
        res = res + e;
    end
    errnorm(3) = sqrt(res);
end

function [ret]=test(x0, beta)
    switch(beta)
        case 1
            ret = psi1(x0);
        case 2
            ret = psi2(x0);
        case 3
            ret = psi3(x0);
        otherwise
            ret = 0;
    end
end

function [ret]=trial(x0, alpha)
    ret = test(x0, alpha);
end

function [ret]=trial_prime(x0, alpha)
    switch(alpha)
        case 1
            ret = psi1_prime(x0);
        case 2
            ret = psi2_prime(x0);
        case 3
            ret = psi3_prime(x0);
        otherwise
            ret = 0;
    end
end

function [ret]=test_prime(x0, beta)
    ret = trial_prime(x0, beta);
end

function [ret]=psi1(x0)
    ret = 2*x0*x0-3*x0+1;
end

function [ret]=psi2(x0)
    ret = 2*x0*x0-x0;
end

function [ret]=psi3(x0)
    ret = -4*x0*x0+4*x0;
end

function [ret]=psi1_prime(x0)
    ret = 4*x0-3;
end

function [ret]=psi2_prime(x0)
    ret = 4*x0-1;
end

function [ret]=psi3_prime(x0)
    ret = -8*x0+4;
end

function [ret]=u(x)
    ret = x*cos(x);
end

function [ret]=u_prime(x)
    ret = cos(x)-x*sin(x);
end

function [ret]=c(x)
    ret = exp(x);
end

function [ret]=f(x)
    ret = -exp(x)*(cos(x)-2*sin(x)-x*cos(x)-x*sin(x));
end
