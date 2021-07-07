clear; close all; clc;

solve_2d_elliptic_pde(2, 2);

function [h1, h2, errnorm] = solve_2d_elliptic_pde(N1, N2)
    x_min = 0.0;
    x_max = 1.0;
    y_min = 0.0;
    y_max = 1.0;
    
    h1 = (x_max-x_min)/N1;
    h2 = (y_max-y_min)/N2;

    [P, T] = mesh_info_mat(x_min,x_max,y_min,y_max,N1,N2);
    [Pb, Tb] = fem_info_mat(x_min,x_max,y_min,y_max,N1,N2);
    [boundary_edge, boundary_node] = boundary_info_mat(N1, N2, T, Tb);
    
    errnorm = 0;

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

function [ret] = psi1(x0, y0)
    ret = 2.0 * (x0 * x0 + y0 * y0) + 4.0 * x0 * y0 - 3.0 * (x0 + y0) + 1.0;
end

function [ret] = psi2(x0, y0)
    ret = x0 * (2.0 * x0 - 1.0);
end

function [ret] = psi3(x0, y0)
    ret = y0 * (2.0 * y0 - 1.0);
end

function [ret] = psi4(x0, y0)
    ret = 4.0 * x0 * (1.0 - x0 - y0);
end

function [ret] = psi5(x0, y0)
    ret = 4.0 * x0 * y0;
end

function [ret] = psi6(x0, y0)
    ret = 4.0 * y0 * (1.0 - x0 - y0);
end
