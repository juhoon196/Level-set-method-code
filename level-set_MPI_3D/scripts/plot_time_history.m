%% plot_time_history.m
%  Scan all binary snapshots, read time from headers,
%  compute L2 error and volume, then plot over time.
%
%  No CSV needed -- everything is read directly from binary files.

clear; clc; close all;

%% Configuration
output_dir = '../output';
if ~exist(output_dir, 'dir'), output_dir = './output'; end
if ~exist(output_dir, 'dir')
    error('Output directory not found.');
end

%% Helper: read field
function [G, nx, ny, nz, nghost, t, dx, dy, dz] = read_field(filepath)
    fid = fopen(filepath, 'rb');
    hi = fread(fid, 4, 'int32');
    nx = hi(1); ny = hi(2); nz = hi(3); nghost = hi(4);
    hd = fread(fid, 4, 'float64');
    t = hd(1); dx = hd(2); dy = hd(3); dz = hd(4);
    nx_t = nx+2*nghost; ny_t = ny+2*nghost; nz_t = nz+2*nghost;
    data = fread(fid, nx_t*ny_t*nz_t, 'float64');
    fclose(fid);
    G_full = reshape(data, [nx_t, ny_t, nz_t]);
    G = G_full(nghost+1:nghost+nx, nghost+1:nghost+ny, nghost+1:nghost+nz);
end

%% Scan files
files_init  = dir(fullfile(output_dir, 'G_initial.bin'));
files_step  = dir(fullfile(output_dir, 'G_step_*.bin'));
files_final = dir(fullfile(output_dir, 'G_final.bin'));

all_files = {};
if ~isempty(files_init)
    all_files{end+1} = fullfile(output_dir, files_init(1).name);
end
if ~isempty(files_step)
    nums = zeros(length(files_step),1);
    for k = 1:length(files_step)
        tok = regexp(files_step(k).name, 'G_step_(\d+)', 'tokens');
        if ~isempty(tok), nums(k) = str2double(tok{1}{1}); end
    end
    [~,si] = sort(nums);
    for k = 1:length(si)
        all_files{end+1} = fullfile(output_dir, files_step(si(k)).name); %#ok<SAGROW>
    end
end
if ~isempty(files_final)
    all_files{end+1} = fullfile(output_dir, files_final(1).name);
end

n = length(all_files);
fprintf('Found %d snapshots. Processing...\n', n);

%% Read initial field as reference
[G_ref, nx, ny, nz, ~, ~, dx, dy, dz] = read_field(all_files{1});

%% Compute L2 error and volume for each snapshot
t_arr   = zeros(n, 1);
l2_arr  = zeros(n, 1);
vol_arr = zeros(n, 1);

for k = 1:n
    [G, ~, ~, ~, ~, t_val, ~, ~, ~] = read_field(all_files{k});
    t_arr(k) = t_val;

    diff = G - G_ref;
    l2_arr(k) = sqrt( mean(diff(:).^2) );

    vol_arr(k) = sum(G(:) < 0) * dx * dy * dz;

    if mod(k, 10) == 0 || k == n
        fprintf('  [%d/%d] t = %.6f\n', k, n, t_val);
    end
end

%% Plot
figure('Name', 'Time History', 'Position', [200 200 1000 500], 'Color', 'w');

subplot(1,2,1);
semilogy(t_arr, l2_arr, 'b-o', 'MarkerSize', 3, 'LineWidth', 1.2);
xlabel('Time'); ylabel('L_2 Error');
title('L_2 Error vs Time'); grid on;
set(gca, 'FontSize', 11);

subplot(1,2,2);
vol0 = vol_arr(1);
vol_pct = 100*(vol_arr - vol0) / vol0;

yyaxis left;
plot(t_arr, vol_arr, 'b-o', 'MarkerSize', 3, 'LineWidth', 1.2);
ylabel('Volume');

yyaxis right;
plot(t_arr, vol_pct, 'r--', 'LineWidth', 1.0);
ylabel('Volume Change (%)');

xlabel('Time');
title('Volume Conservation'); grid on;
legend('Volume', 'Change (%)', 'Location', 'best');
set(gca, 'FontSize', 11);

sgtitle('Level-Set 3D Simulation Summary', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('\nInitial volume: %.6f\n', vol0);
fprintf('Final volume:   %.6f\n', vol_arr(end));
fprintf('Volume change:  %.4f%%\n', vol_pct(end));
fprintf('Final L2 error: %.6e\n', l2_arr(end));

end
