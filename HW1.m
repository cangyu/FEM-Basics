function [h, errnorm] = HW1(N)

xa = 0.0;
xb = 1.0;

h = (xb-xa) / N; h2 = h*h;
Nm = N + 1;

P = zeros(1, Nm);
for j = 1:Nm
    P(j) = xa + (j-1) * h;
end

T = zeros(2, N);
for j = 1:N
    T(1, j) = j;
    T(2, j) = j+1;
end

Nb = N + 1;

Pb = zeros(1, Nb);
for j = 1:Nb
    Pb(j) = xa + (j-1) * h;
end

Tb = zeros(2, N);
for j = 1:N
    Tb(1, j) = j;
    Tb(2, j) = j+1;
end

Nlb_trial = 2;
Nlb_test = 2;

A = zeros(Nm, Nm);
b = zeros(Nm, 1);

gauss_quad_s=[-sqrt(3/5),0,sqrt(3/5)];
gauss_quad_w=[5/9, 8/9, 5/9];
gauss_quad_n=3;

for n = 1:N
    xn0 = (n-1)*h; xn1 = n*h;
    
    % gauss quadrature points
    gauss_quad_x=zeros(1,gauss_quad_n);
    for i = 1:gauss_quad_n
        gauss_quad_x(i) = ((xn1+xn0)+(xn1-xn0)*gauss_quad_s(i))/2;
    end
    
    % coefficient matrix
    for alpha = 1:Nlb_trial
        for beta = 1:Nlb_test
            r = (exp(xn1) - exp(xn0))/h2 * power(-1, alpha+beta);
            gi = Tb(beta, n); gj = Tb(alpha, n);
            A(gi, gj) = A(gi, gj) + r;
        end
    end
    
    % load vector
    for beta = 1:Nlb_test
        r = 0;
        for i = 1:gauss_quad_n
            r = r + gauss_quad_w(i) * f(gauss_quad_x(i)) * test(xn0, xn1, gauss_quad_x(i), beta);
        end
        r = r * (xn1-xn0)/2;
        gi = Tb(beta, n);
        b(gi) = b(gi) + r;
    end
end

% boundary condition
A(1,:) = 0;
A(1,1) = 1;
b(1) = u(xa);

A(Nm,:) = 0;
A(Nm,Nm) = 1;
b(Nm) = u(xb);

% solve
sol = A \ b;

% check
err = zeros(Nm, 1);
for i = 1:Nm
    err(i) = u(P(i)) - sol(i);
end

errnorm = max(abs(err));
end

function [ret]=test(a, b, x, beta)
    h = b-a;
    ra = (b-x)/h;
    rb = (x-a)/h;
    ret = (1-power(-1, beta))/2 * ra + (1+power(-1, beta))/2 * rb;
end

function [ret]=u(x)
    ret = x*cos(x);
end

function [ret]=f(x)
    ret = -exp(x)*(cos(x)-2*sin(x)-x*cos(x)-x*sin(x));
end
