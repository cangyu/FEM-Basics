clear; close all; clc;

solve_2d_elliptic_pde(3, 3);

function [h1, h2, errnorm] = solve_2d_elliptic_pde(N1, N2)
    x_min = 0.0;
    x_max = 1.0;
    y_min = 0.0;
    y_max = 1.0;
    
    h1 = (x_max-x_min)/N1;
    h2 = (y_max-y_min)/N2;

    [P, T] = mesh_info_mat(x_min,x_max,y_min,y_max,N1,N2);
    
    Pb = P;
    Tb = T;
        
    [boundary_edge, boundary_node] = boundary_info_mat(N1, N2, T);
    
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
