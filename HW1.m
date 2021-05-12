clear; close all; clc;

xa = 0.0;
xb = 1.0;
N = 8;

h = (xb-xa) / N;
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

A = zeros(Nm, Nm);
b = zeros(Nm, 1);

