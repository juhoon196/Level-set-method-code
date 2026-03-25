function [G, nx, ny, nghost, x, y] = read_level_set_binary(filename)
% READ_LEVEL_SET_BINARY Read binary output from G-equation Level-Set solver
%
% Syntax:
%   [G, nx, ny, nghost] = read_level_set_binary(filename)
%   [G, nx, ny, nghost, x, y] = read_level_set_binary(filename)
%
% Input:
%   filename - Path to binary file (.bin)
%
% Output:
%   G      - 2D array of level-set values (ny x nx), interior points only
%   nx, ny - Interior grid dimensions
%   nghost - Number of ghost cells
%   x, y   - Coordinate arrays (optional)
%
% File format:
%   Header: nx, ny, nghost (3 x int32)
%   Data: G values (float64, row-major order including ghost cells)
%
% Example:
%   [G, nx, ny, ~, x, y] = read_level_set_binary('output/G_final.bin');
%   contourf(x, y, G, 30);
%   colorbar;
%
% See also: visualize, analyze_error

    % Open file
    fid = fopen(filename, 'rb');
    if fid == -1
        error('read_level_set_binary:FileNotFound', ...
              'Cannot open file: %s', filename);
    end

    % Read header
    header = fread(fid, 3, 'int32');
    if length(header) < 3
        fclose(fid);
        error('read_level_set_binary:InvalidHeader', ...
              'Invalid file header in: %s', filename);
    end

    nx = header(1);
    ny = header(2);
    nghost = header(3);

    % Calculate total size with ghost cells
    nx_total = nx + 2*nghost;
    ny_total = ny + 2*nghost;
    total_size = nx_total * ny_total;

    % Read data
    data = fread(fid, total_size, 'float64');
    fclose(fid);

    if length(data) < total_size
        error('read_level_set_binary:IncompleteData', ...
              'File contains incomplete data: expected %d, got %d values', ...
              total_size, length(data));
    end

    % Reshape to 2D array
    % Data is stored in row-major order (C-style)
    % MATLAB uses column-major, so we need to reshape and transpose
    G_full = reshape(data, [nx_total, ny_total])';

    % Extract interior points (remove ghost cells)
    G = G_full(nghost+1:nghost+ny, nghost+1:nghost+nx);

    % Generate coordinate arrays if requested
    if nargout > 4
        x = linspace(0, 1, nx);
        y = linspace(0, 1, ny);
    end
end
