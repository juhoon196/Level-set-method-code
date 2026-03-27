%% compare_reinit.m
% Compare results with and without reinitialization
%
% This script compares simulation results from two runs:
% 1. With HCR-2 reinitialization enabled
% 2. Without reinitialization
%
% Usage:
%   1. Run solver with reinitialization, save results to output_reinit/
%   2. Run solver without reinitialization, save results to output_no_reinit/
%   3. Run this script
%
% Commands to generate data:
%   ./g_equation_solver -t pyramid -T 1.0 -reinit -o output_reinit
%   ./g_equation_solver -t pyramid -T 1.0 -no-reinit -o output_no_reinit

clear; close all; clc;

%% Configuration
dir_reinit = '../output_reinit';
dir_no_reinit = '../output_no_reinit';

% Check if directories exist
if ~exist(dir_reinit, 'dir') || ~exist(dir_no_reinit, 'dir')
    fprintf('Data directories not found.\n');
    fprintf('Please run the following commands first:\n\n');
    fprintf('  cd /path/to/level-set\n');
    fprintf('  ./g_equation_solver -t pyramid -T 1.0 -reinit -o output_reinit\n');
    fprintf('  ./g_equation_solver -t pyramid -T 1.0 -no-reinit -o output_no_reinit\n\n');

    % Use default output directory as fallback
    fprintf('Using default output directory for single comparison...\n');
    dir_reinit = '../output';
    dir_no_reinit = '../output';
end

%% Load Data
fprintf('Loading data...\n');

% Initial condition (should be same for both)
[G_initial, nx, ny, nghost, x, y] = read_level_set_binary(...
    fullfile(dir_reinit, 'G_initial.bin'));

% Final results
[G_reinit, ~, ~, ~] = read_level_set_binary(fullfile(dir_reinit, 'G_final.bin'));

if ~strcmp(dir_reinit, dir_no_reinit)
    [G_no_reinit, ~, ~, ~] = read_level_set_binary(fullfile(dir_no_reinit, 'G_final.bin'));
else
    G_no_reinit = G_reinit;  % Same file, just for demonstration
    fprintf('Note: Using same data for both (run two simulations for real comparison)\n\n');
end

dx = 1.0 / (nx - 1);
[X, Y] = meshgrid(x, y);

%% Calculate Error Metrics
fprintf('=== Error Comparison ===\n\n');

% With reinitialization
diff_reinit = G_reinit - G_initial;
L2_reinit = sqrt(mean(diff_reinit(:).^2));
Linf_reinit = max(abs(diff_reinit(:)));
area_init = sum(G_initial(:) < 0);
area_reinit = sum(G_reinit(:) < 0);
area_err_reinit = abs(area_reinit - area_init) / area_init * 100;

% Without reinitialization
diff_no_reinit = G_no_reinit - G_initial;
L2_no_reinit = sqrt(mean(diff_no_reinit(:).^2));
Linf_no_reinit = max(abs(diff_no_reinit(:)));
area_no_reinit = sum(G_no_reinit(:) < 0);
area_err_no_reinit = abs(area_no_reinit - area_init) / area_init * 100;

% Gradient magnitude (SDF property)
[Gx_r, Gy_r] = gradient(G_reinit, dx, dx);
[Gx_nr, Gy_nr] = gradient(G_no_reinit, dx, dx);
grad_reinit = sqrt(Gx_r.^2 + Gy_r.^2);
grad_no_reinit = sqrt(Gx_nr.^2 + Gy_nr.^2);

near_interface_r = abs(G_reinit) < 3*dx;
near_interface_nr = abs(G_no_reinit) < 3*dx;

grad_mean_reinit = mean(grad_reinit(near_interface_r));
grad_mean_no_reinit = mean(grad_no_reinit(near_interface_nr));

%% Print Results Table
fprintf('                        With Reinit    Without Reinit\n');
fprintf('--------------------------------------------------------\n');
fprintf('L2 Error                %.4e      %.4e\n', L2_reinit, L2_no_reinit);
fprintf('L_inf Error             %.4e      %.4e\n', Linf_reinit, Linf_no_reinit);
fprintf('Area Error              %.2f%%          %.2f%%\n', area_err_reinit, area_err_no_reinit);
fprintf('|∇G| Mean (interface)   %.4f          %.4f\n', grad_mean_reinit, grad_mean_no_reinit);
fprintf('--------------------------------------------------------\n');

%% Visualization
figure('Position', [100, 100, 1400, 900], 'Color', 'w');

% Row 1: Level-set fields
subplot(2,3,1);
contourf(X, Y, G_initial, 30, 'LineStyle', 'none');
hold on;
contour(X, Y, G_initial, [0 0], 'k', 'LineWidth', 2);
hold off;
colorbar; axis equal tight;
title('Initial'); xlabel('x'); ylabel('y');

subplot(2,3,2);
contourf(X, Y, G_reinit, 30, 'LineStyle', 'none');
hold on;
contour(X, Y, G_reinit, [0 0], 'k', 'LineWidth', 2);
hold off;
colorbar; axis equal tight;
title(sprintf('With Reinit (L_2=%.2e)', L2_reinit)); xlabel('x'); ylabel('y');

subplot(2,3,3);
contourf(X, Y, G_no_reinit, 30, 'LineStyle', 'none');
hold on;
contour(X, Y, G_no_reinit, [0 0], 'k', 'LineWidth', 2);
hold off;
colorbar; axis equal tight;
title(sprintf('Without Reinit (L_2=%.2e)', L2_no_reinit)); xlabel('x'); ylabel('y');

% Row 2: Error and gradient analysis
subplot(2,3,4);
contour(X, Y, G_initial, [0 0], 'b-', 'LineWidth', 2);
hold on;
contour(X, Y, G_reinit, [0 0], 'r--', 'LineWidth', 2);
contour(X, Y, G_no_reinit, [0 0], 'g:', 'LineWidth', 2);
hold off;
axis equal; xlim([0 1]); ylim([0 1]); grid on;
legend('Initial', 'Reinit', 'No Reinit', 'Location', 'best');
title('Interface Comparison'); xlabel('x'); ylabel('y');

subplot(2,3,5);
imagesc(x, y, grad_reinit);
colorbar; caxis([0, 2]);
hold on;
contour(X, Y, G_reinit, [0 0], 'w', 'LineWidth', 1.5);
hold off;
axis equal tight;
title(sprintf('|∇G| With Reinit (mean=%.2f)', grad_mean_reinit));
xlabel('x'); ylabel('y');

subplot(2,3,6);
imagesc(x, y, grad_no_reinit);
colorbar; caxis([0, 2]);
hold on;
contour(X, Y, G_no_reinit, [0 0], 'w', 'LineWidth', 1.5);
hold off;
axis equal tight;
title(sprintf('|∇G| Without Reinit (mean=%.2f)', grad_mean_no_reinit));
xlabel('x'); ylabel('y');

sgtitle('Reinitialization Comparison', 'FontSize', 14, 'FontWeight', 'bold');

% Save figure
saveas(gcf, fullfile(dir_reinit, 'reinit_comparison.png'));
fprintf('\nSaved: %s\n', fullfile(dir_reinit, 'reinit_comparison.png'));

%% Bar chart comparison
figure('Position', [100, 100, 800, 400], 'Color', 'w');

metrics = {'L_2 Error', 'L_\infty Error', 'Area Error (%)', '||∇G|-1|'};
values_reinit = [L2_reinit, Linf_reinit, area_err_reinit, abs(grad_mean_reinit-1)];
values_no_reinit = [L2_no_reinit, Linf_no_reinit, area_err_no_reinit, abs(grad_mean_no_reinit-1)];

X_bar = categorical(metrics);
X_bar = reordercats(X_bar, metrics);

bar(X_bar, [values_reinit; values_no_reinit]');
legend('With Reinit', 'Without Reinit', 'Location', 'best');
ylabel('Error Value');
title('Error Metrics Comparison');
grid on;

saveas(gcf, fullfile(dir_reinit, 'error_bar_chart.png'));
fprintf('Saved: %s\n', fullfile(dir_reinit, 'error_bar_chart.png'));

fprintf('\nComparison complete!\n');
