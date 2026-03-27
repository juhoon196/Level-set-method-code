%% visualize.m
% Visualization script for G-equation Level-Set Solver results
%
% This script reads binary output files and creates:
% - Contour plots of the level-set field
% - Comparison between initial and final states
% - Error analysis
%
% Usage:
%   Run this script from MATLAB after setting the output_dir path

clear; close all; clc;

%% Configuration
output_dir = '../output';  % Adjust path as needed

% File names
initial_file = fullfile(output_dir, 'G_initial.bin');
final_file = fullfile(output_dir, 'G_final.bin');

%% Read Binary Files
fprintf('Loading data files...\n');

[G_initial, nx, ny, nghost] = read_level_set_binary(initial_file);
[G_final, ~, ~, ~] = read_level_set_binary(final_file);

fprintf('Grid size: %d x %d\n', nx, ny);
fprintf('Ghost cells: %d\n', nghost);

%% Create coordinate arrays
x = linspace(0, 1, nx);
y = linspace(0, 1, ny);
[X, Y] = meshgrid(x, y);

%% Calculate Error Metrics
diff = G_final - G_initial;
L2_error = sqrt(mean(diff(:).^2));
Linf_error = max(abs(diff(:)));

% Area calculation (cells where G < 0)
initial_area = sum(G_initial(:) < 0) / (nx * ny);
final_area = sum(G_final(:) < 0) / (nx * ny);
area_error = abs(final_area - initial_area) / initial_area * 100;

fprintf('\n=== Error Metrics ===\n');
fprintf('L2 Error:           %.6e\n', L2_error);
fprintf('L_inf Error:        %.6e\n', Linf_error);
fprintf('Initial Area:       %.6f\n', initial_area);
fprintf('Final Area:         %.6f\n', final_area);
fprintf('Area Error:         %.2f%%\n', area_error);

%% Figure 1: Side-by-side comparison
figure('Position', [100, 100, 1500, 450], 'Color', 'w');

% Initial state
subplot(1,3,1);
contourf(X, Y, G_initial, 30, 'LineStyle', 'none');
hold on;
contour(X, Y, G_initial, [0 0], 'k', 'LineWidth', 2);
hold off;
colorbar;
colormap(bluewhitered);
caxis([-max(abs(G_initial(:))), max(abs(G_initial(:)))]);
axis equal tight;
xlabel('x'); ylabel('y');
title('Initial State (t = 0)');
set(gca, 'FontSize', 12);

% Final state
subplot(1,3,2);
contourf(X, Y, G_final, 30, 'LineStyle', 'none');
hold on;
contour(X, Y, G_final, [0 0], 'k', 'LineWidth', 2);
hold off;
colorbar;
colormap(bluewhitered);
caxis([-max(abs(G_final(:))), max(abs(G_final(:)))]);
axis equal tight;
xlabel('x'); ylabel('y');
title('Final State (t = T)');
set(gca, 'FontSize', 12);

% Difference
subplot(1,3,3);
contourf(X, Y, diff, 30, 'LineStyle', 'none');
hold on;
contour(X, Y, G_initial, [0 0], 'b--', 'LineWidth', 1.5);
contour(X, Y, G_final, [0 0], 'r-', 'LineWidth', 1.5);
hold off;
colorbar;
colormap(bluewhitered);
caxis([-max(abs(diff(:))), max(abs(diff(:)))]);
axis equal tight;
xlabel('x'); ylabel('y');
title(sprintf('Difference (L_2 = %.2e)', L2_error));
legend('', 'Initial', 'Final', 'Location', 'northeast');
set(gca, 'FontSize', 12);

sgtitle('G-equation Level-Set Solver Results', 'FontSize', 14, 'FontWeight', 'bold');

% Save figure
saveas(gcf, fullfile(output_dir, 'comparison.png'));
fprintf('\nSaved: %s\n', fullfile(output_dir, 'comparison.png'));

%% Figure 2: Interface comparison only
figure('Position', [100, 600, 600, 550], 'Color', 'w');

contour(X, Y, G_initial, [0 0], 'b--', 'LineWidth', 2);
hold on;
contour(X, Y, G_final, [0 0], 'r-', 'LineWidth', 2);
hold off;

axis equal;
xlim([0 1]); ylim([0 1]);
grid on;
xlabel('x'); ylabel('y');
title('Interface Comparison (G = 0)');
legend('Initial (t = 0)', 'Final (t = T)', 'Location', 'northeast');
set(gca, 'FontSize', 12);

% Save figure
saveas(gcf, fullfile(output_dir, 'interface_comparison.png'));
fprintf('Saved: %s\n', fullfile(output_dir, 'interface_comparison.png'));

%% Figure 3: 3D surface plot
figure('Position', [750, 600, 600, 550], 'Color', 'w');

surf(X, Y, G_final, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
hold on;
contour3(X, Y, G_final, [0 0], 'k', 'LineWidth', 2);
hold off;

colorbar;
colormap(bluewhitered);
xlabel('x'); ylabel('y'); zlabel('G');
title('Final Level-Set Field (3D)');
view([-30, 30]);
set(gca, 'FontSize', 12);

% Save figure
saveas(gcf, fullfile(output_dir, 'surface_3d.png'));
fprintf('Saved: %s\n', fullfile(output_dir, 'surface_3d.png'));

%% Figure 4: Cross-section comparison
figure('Position', [100, 100, 1000, 400], 'Color', 'w');

% Find center index
j_center = round(ny/2);

subplot(1,2,1);
plot(x, G_initial(j_center, :), 'b-', 'LineWidth', 2);
hold on;
plot(x, G_final(j_center, :), 'r--', 'LineWidth', 2);
yline(0, 'k:', 'LineWidth', 1);
hold off;
xlabel('x'); ylabel('G');
title(sprintf('Cross-section at y = %.2f', y(j_center)));
legend('Initial', 'Final', 'Interface', 'Location', 'best');
grid on;
set(gca, 'FontSize', 12);

subplot(1,2,2);
plot(x, G_initial(j_center, :) - G_final(j_center, :), 'k-', 'LineWidth', 2);
xlabel('x'); ylabel('\Delta G');
title('Difference along cross-section');
grid on;
set(gca, 'FontSize', 12);

% Save figure
saveas(gcf, fullfile(output_dir, 'cross_section.png'));
fprintf('Saved: %s\n', fullfile(output_dir, 'cross_section.png'));

fprintf('\nVisualization complete!\n');

%% ========================================================================
%  Helper Functions
%  ========================================================================

function [G, nx, ny, nghost] = read_level_set_binary(filename)
    % READ_LEVEL_SET_BINARY Read binary output from G-equation solver
    %
    % File format:
    %   Header: nx, ny, nghost (3 x int32)
    %   Data: G values (float64, row-major order with ghost cells)
    %
    % Returns:
    %   G: 2D array of interior points only (ny x nx)
    %   nx, ny: Interior grid dimensions
    %   nghost: Number of ghost cells

    fid = fopen(filename, 'rb');
    if fid == -1
        error('Cannot open file: %s', filename);
    end

    % Read header
    header = fread(fid, 3, 'int32');
    nx = header(1);
    ny = header(2);
    nghost = header(3);

    % Calculate total size
    nx_total = nx + 2*nghost;
    ny_total = ny + 2*nghost;

    % Read data
    data = fread(fid, nx_total * ny_total, 'float64');
    fclose(fid);

    % Reshape to 2D (row-major to column-major conversion)
    G_full = reshape(data, [nx_total, ny_total])';

    % Extract interior points (remove ghost cells)
    G = G_full(nghost+1:nghost+ny, nghost+1:nghost+nx);
end

function c = bluewhitered(m)
    % BLUEWHITERED Blue-white-red colormap for diverging data
    %
    % Creates a colormap that goes from blue (negative) through white (zero)
    % to red (positive). Useful for visualizing signed data.

    if nargin < 1
        m = 256;
    end

    % Define key colors
    bottom = [0, 0, 0.5];      % Dark blue
    middle_low = [0, 0.5, 1];  % Light blue
    middle = [1, 1, 1];        % White
    middle_high = [1, 0.5, 0]; % Orange
    top = [0.5, 0, 0];         % Dark red

    % Create colormap
    n = ceil(m/4);

    c1 = [linspace(bottom(1), middle_low(1), n)', ...
          linspace(bottom(2), middle_low(2), n)', ...
          linspace(bottom(3), middle_low(3), n)'];

    c2 = [linspace(middle_low(1), middle(1), n)', ...
          linspace(middle_low(2), middle(2), n)', ...
          linspace(middle_low(3), middle(3), n)'];

    c3 = [linspace(middle(1), middle_high(1), n)', ...
          linspace(middle(2), middle_high(2), n)', ...
          linspace(middle(3), middle_high(3), n)'];

    c4 = [linspace(middle_high(1), top(1), n)', ...
          linspace(middle_high(2), top(2), n)', ...
          linspace(middle_high(3), top(3), n)'];

    c = [c1; c2(2:end,:); c3(2:end,:); c4(2:end,:)];

    % Trim or extend to exact size
    if size(c,1) > m
        c = c(1:m, :);
    elseif size(c,1) < m
        c = interp1(1:size(c,1), c, linspace(1, size(c,1), m));
    end
end
