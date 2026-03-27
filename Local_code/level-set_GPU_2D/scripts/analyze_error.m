%% analyze_error.m
% Error analysis script for G-equation Level-Set Solver
%
% This script performs detailed error analysis:
% - L2 and L_inf error norms
% - Interface position error
% - Area/mass conservation
% - Gradient magnitude analysis (SDF property)

clear; close all; clc;

%% Configuration
output_dir = '../output';

initial_file = fullfile(output_dir, 'G_initial.bin');
final_file = fullfile(output_dir, 'G_final.bin');

%% Read Data
fprintf('=== G-equation Level-Set Solver Error Analysis ===\n\n');

[G_initial, nx, ny, nghost] = read_level_set_binary(initial_file);
[G_final, ~, ~, ~] = read_level_set_binary(final_file);

dx = 1.0 / (nx - 1);
dy = 1.0 / (ny - 1);

x = linspace(0, 1, nx);
y = linspace(0, 1, ny);
[X, Y] = meshgrid(x, y);

%% 1. Global Error Metrics
fprintf('1. Global Error Metrics\n');
fprintf('   ---------------------\n');

diff = G_final - G_initial;

% L2 error (RMS)
L2_error = sqrt(mean(diff(:).^2));

% L_inf error (max absolute)
Linf_error = max(abs(diff(:)));

% Relative L2 error
L2_rel = L2_error / sqrt(mean(G_initial(:).^2));

fprintf('   L2 Error (absolute):    %.6e\n', L2_error);
fprintf('   L_inf Error:            %.6e\n', Linf_error);
fprintf('   L2 Error (relative):    %.4f%%\n', L2_rel * 100);

%% 2. Interface Position Error
fprintf('\n2. Interface Position Error\n');
fprintf('   -------------------------\n');

% Extract zero contours
C_initial = contourc(x, y, G_initial, [0 0]);
C_final = contourc(x, y, G_final, [0 0]);

% Parse contour data
[xi, yi] = parse_contour(C_initial);
[xf, yf] = parse_contour(C_final);

% Calculate Hausdorff distance (approximate)
if ~isempty(xi) && ~isempty(xf)
    d_init_to_final = zeros(length(xi), 1);
    for i = 1:length(xi)
        distances = sqrt((xf - xi(i)).^2 + (yf - yi(i)).^2);
        d_init_to_final(i) = min(distances);
    end

    d_final_to_init = zeros(length(xf), 1);
    for i = 1:length(xf)
        distances = sqrt((xi - xf(i)).^2 + (yi - yf(i)).^2);
        d_final_to_init(i) = min(distances);
    end

    hausdorff = max(max(d_init_to_final), max(d_final_to_init));
    mean_interface_error = mean([d_init_to_final; d_final_to_init]);

    fprintf('   Hausdorff distance:     %.6e\n', hausdorff);
    fprintf('   Mean interface error:   %.6e\n', mean_interface_error);
    fprintf('   (in grid cells):        %.2f dx\n', mean_interface_error / dx);
else
    fprintf('   Could not extract interface contours\n');
end

%% 3. Area/Mass Conservation
fprintf('\n3. Area Conservation\n');
fprintf('   ------------------\n');

% Method 1: Cell counting
area_init_cells = sum(G_initial(:) < 0);
area_final_cells = sum(G_final(:) < 0);
area_error_cells = abs(area_final_cells - area_init_cells) / area_init_cells * 100;

fprintf('   Initial area (cells):   %d\n', area_init_cells);
fprintf('   Final area (cells):     %d\n', area_final_cells);
fprintf('   Area change:            %+d cells (%.2f%%)\n', ...
        area_final_cells - area_init_cells, area_error_cells);

% Method 2: Heaviside integration
area_init_H = sum(sum(heaviside_smooth(-G_initial, dx))) * dx * dy;
area_final_H = sum(sum(heaviside_smooth(-G_final, dx))) * dx * dy;
area_error_H = abs(area_final_H - area_init_H) / area_init_H * 100;

fprintf('   Initial area (smooth):  %.6f\n', area_init_H);
fprintf('   Final area (smooth):    %.6f\n', area_final_H);
fprintf('   Area error (smooth):    %.4f%%\n', area_error_H);

%% 4. Signed Distance Function Property
fprintf('\n4. SDF Property (|grad G| = 1)\n');
fprintf('   ----------------------------\n');

% Compute gradient magnitude
[Gx_init, Gy_init] = gradient(G_initial, dx, dy);
grad_mag_init = sqrt(Gx_init.^2 + Gy_init.^2);

[Gx_final, Gy_final] = gradient(G_final, dx, dy);
grad_mag_final = sqrt(Gx_final.^2 + Gy_final.^2);

% Analyze near interface (|G| < 3*dx)
near_interface_init = abs(G_initial) < 3*dx;
near_interface_final = abs(G_final) < 3*dx;

grad_error_init = abs(grad_mag_init - 1);
grad_error_final = abs(grad_mag_final - 1);

fprintf('   Initial |grad G| near interface:\n');
fprintf('      Mean:  %.4f (ideal: 1.0)\n', mean(grad_mag_init(near_interface_init)));
fprintf('      Std:   %.4f\n', std(grad_mag_init(near_interface_init)));

fprintf('   Final |grad G| near interface:\n');
fprintf('      Mean:  %.4f (ideal: 1.0)\n', mean(grad_mag_final(near_interface_final)));
fprintf('      Std:   %.4f\n', std(grad_mag_final(near_interface_final)));

%% 5. Visualization
figure('Position', [100, 100, 1400, 800], 'Color', 'w');

% Error distribution
subplot(2,3,1);
histogram(diff(:), 50, 'Normalization', 'pdf');
xlabel('\Delta G'); ylabel('PDF');
title('Error Distribution');
xline(0, 'r--', 'LineWidth', 1.5);
grid on;

% Error spatial distribution
subplot(2,3,2);
imagesc(x, y, abs(diff));
colorbar;
axis equal tight;
xlabel('x'); ylabel('y');
title('|G_{final} - G_{initial}|');
colormap(gca, 'hot');

% Gradient magnitude (initial)
subplot(2,3,3);
imagesc(x, y, grad_mag_init);
colorbar;
hold on;
contour(X, Y, G_initial, [0 0], 'w', 'LineWidth', 1.5);
hold off;
axis equal tight;
xlabel('x'); ylabel('y');
title('|∇G| Initial');
caxis([0, 2]);

% Gradient magnitude (final)
subplot(2,3,4);
imagesc(x, y, grad_mag_final);
colorbar;
hold on;
contour(X, Y, G_final, [0 0], 'w', 'LineWidth', 1.5);
hold off;
axis equal tight;
xlabel('x'); ylabel('y');
title('|∇G| Final');
caxis([0, 2]);

% Gradient error near interface
subplot(2,3,5);
grad_error_masked = grad_error_final;
grad_error_masked(~near_interface_final) = NaN;
imagesc(x, y, grad_error_masked);
colorbar;
hold on;
contour(X, Y, G_final, [0 0], 'k', 'LineWidth', 1.5);
hold off;
axis equal tight;
xlabel('x'); ylabel('y');
title('||∇G| - 1| (near interface)');

% Interface comparison
subplot(2,3,6);
contour(X, Y, G_initial, [0 0], 'b-', 'LineWidth', 2);
hold on;
contour(X, Y, G_final, [0 0], 'r--', 'LineWidth', 2);
hold off;
axis equal;
xlim([0 1]); ylim([0 1]);
grid on;
xlabel('x'); ylabel('y');
title('Interface: Initial (blue) vs Final (red)');

sgtitle('Error Analysis', 'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, fullfile(output_dir, 'error_analysis.png'));
fprintf('\nSaved: %s\n', fullfile(output_dir, 'error_analysis.png'));

%% Summary Table
fprintf('\n========================================\n');
fprintf('           SUMMARY TABLE\n');
fprintf('========================================\n');
fprintf('Metric                    Value\n');
fprintf('----------------------------------------\n');
fprintf('L2 Error                  %.4e\n', L2_error);
fprintf('L_inf Error               %.4e\n', Linf_error);
fprintf('Area Conservation         %.2f%%\n', 100 - area_error_cells);
fprintf('|∇G| Mean (final)         %.4f\n', mean(grad_mag_final(near_interface_final)));
fprintf('========================================\n');

%% ========================================================================
%  Helper Functions
%  ========================================================================

function [G, nx, ny, nghost] = read_level_set_binary(filename)
    fid = fopen(filename, 'rb');
    if fid == -1
        error('Cannot open file: %s', filename);
    end

    header = fread(fid, 3, 'int32');
    nx = header(1);
    ny = header(2);
    nghost = header(3);

    nx_total = nx + 2*nghost;
    ny_total = ny + 2*nghost;

    data = fread(fid, nx_total * ny_total, 'float64');
    fclose(fid);

    G_full = reshape(data, [nx_total, ny_total])';
    G = G_full(nghost+1:nghost+ny, nghost+1:nghost+nx);
end

function [xc, yc] = parse_contour(C)
    % Parse MATLAB contour matrix format
    xc = [];
    yc = [];

    if isempty(C)
        return;
    end

    idx = 1;
    while idx < size(C, 2)
        n = C(2, idx);
        xc = [xc; C(1, idx+1:idx+n)'];
        yc = [yc; C(2, idx+1:idx+n)'];
        idx = idx + n + 1;
    end
end

function H = heaviside_smooth(phi, epsilon)
    % Smoothed Heaviside function
    H = zeros(size(phi));
    H(phi > epsilon) = 1;
    H(phi < -epsilon) = 0;
    mask = abs(phi) <= epsilon;
    H(mask) = 0.5 * (1 + phi(mask)/epsilon + sin(pi*phi(mask)/epsilon)/pi);
end
