%% animate_results.m
clear; 
% close all;
clc;

%% Configuration
data_dir = '../output'; 
file_pattern = fullfile(data_dir, 'G_step_*.bin');
files = dir(file_pattern);

filenames = {files.name};
[~, idx] = sort(filenames);
sorted_files = filenames(idx);

%% Initialization
first_file = fullfile(data_dir, sorted_files{1});
[G_data, nx, ny, ~, x, y] = read_level_set_binary(first_file);
[X, Y] = meshgrid(x, y);

% %% Animation Loop
% for i = 1:length(sorted_files)
%     current_file = fullfile(data_dir, sorted_files{i});
%     [G, ~, ~, ~, ~, ~] = read_level_set_binary(current_file);
% 
%     contour(X, Y, G, [0 0], 'LineWidth', 2);
%     axis equal tight;
%     grid on;
%     drawnow;
% end

%% Final Comparison Plot (Added)
% Load Initial and Final data
[G_init, ~, ~, ~, ~, ~] = read_level_set_binary(fullfile(data_dir, 'G_initial.bin'));
[G_final, ~, ~, ~, ~, ~] = read_level_set_binary(fullfile(data_dir, 'G_final.bin'));

% Calculation of L2 Error for quantitative check
l2_error = sqrt(mean((G_final(:) - G_init(:)).^2));
fprintf('Final L2 Error: %.4e\n', l2_error);

% Overlay Initial vs Final shapes
figure;
hold on;
% Black solid line for Initial shape (Exact solution after 1 period)
contour(X, Y, G_init, [0 0], 'k-', 'LineWidth', 2); 
% Red dashed line for Final shape (Computed result)
contour(X, Y, G_final, [0 0], 'r--', 'LineWidth', 2);
% hold off;

axis equal tight;
grid on;
legend('Initial (Exact)', 'Final (Result)');