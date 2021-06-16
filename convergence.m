clear; close all; clc;

N_CASE = 9;
h = zeros(1, N_CASE);
err = zeros(1, N_CASE);

for i = 1:N_CASE
    N = 2^i;
    [ch, ce] = HW2(N);
    h(i) = ch;
    err(i) = ce;
    fprintf("h=1/%d, |err|=%e\n", N, ce);
end

loglog(h, err, '-s')
grid on