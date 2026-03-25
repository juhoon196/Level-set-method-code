%% compare_snapshots.m
%  Side-by-side comparison: shape evolution, cross-sections, initial vs final.
%  Reads time directly from the binary file headers (no CSV needed).

clear; clc; close all;

%% Configuration
output_dir = '../output';
if ~exist(output_dir, 'dir'), output_dir = './output'; end
if ~exist(output_dir, 'dir')
    error('Output directory not found. Adjust output_dir.');
end

iso_value = 0.0;

%% Helper: read binary field
function [G, nx, ny, nz, nghost, t, xv, yv, zv] = read_field(filepath)
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
    xv = (0:nx-1)*dx;  yv = (0:ny-1)*dy;  zv = (0:nz-1)*dz;
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

n_snap = length(all_files);
fprintf('Found %d snapshots.\n', n_snap);

%% Read times
times = zeros(n_snap,1);
for k = 1:n_snap
    fid = fopen(all_files{k}, 'rb');
    fread(fid, 4, 'int32');
    hd = fread(fid, 4, 'float64');
    times(k) = hd(1);
    fclose(fid);
end

%% Select up to 6 evenly spaced snapshots
n_panels = min(6, n_snap);
if n_snap <= n_panels
    sel = 1:n_snap;
else
    sel = round(linspace(1, n_snap, n_panels));
end

%% ========================================================================
%  Figure 1: Isosurface at selected times
%  ========================================================================
figure('Name', 'Shape Evolution', 'Position', [50 100 1400 700], 'Color', 'w');

for p = 1:length(sel)
    [G, nx, ny, nz, ~, t_val, xv, yv, zv] = read_field(all_files{sel(p)});
    [X, Y, Z] = meshgrid(xv, yv, zv);
    Gp = permute(G, [2 1 3]);

    subplot(2, 3, p);
    pp = patch(isosurface(X, Y, Z, Gp, iso_value));
    isonormals(X, Y, Z, Gp, pp);
    set(pp, 'FaceColor', [0.2 0.6 1.0], 'EdgeColor', 'none', ...
            'FaceAlpha', 0.7, 'FaceLighting', 'gouraud');
    light('Position', [1 1 1]);
    axis equal; xlim([xv(1) xv(end)]); ylim([yv(1) yv(end)]); zlim([zv(1) zv(end)]);
    view(35, 25); grid on;
    title(sprintf('t = %.4f', t_val), 'FontSize', 12);
    xlabel('X'); ylabel('Y'); zlabel('Z');
end
sgtitle('Shape Evolution (G = 0 isosurface)', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  Figure 2: Cross-section slices (z = mid-plane)
%  ========================================================================
figure('Name', 'Cross Sections (z mid)', 'Position', [50 50 1400 700], 'Color', 'w');

for p = 1:length(sel)
    [G, ~, ~, ~, ~, t_val, xv, yv, zv] = read_field(all_files{sel(p)});
    [~, k_mid] = min(abs(zv - 0.5*(zv(1)+zv(end))));
    G_slice = squeeze(G(:,:,k_mid))';  % (y, x) for plotting

    subplot(2, 3, p);
    contourf(xv, yv, G_slice, [-1e6, 0], 'LineStyle', 'none');
    colormap(gca, [0.85 0.92 1.0; 1 1 1]);
    hold on;
    contour(xv, yv, G_slice, [0 0], 'LineWidth', 2, 'LineColor', [0.1 0.3 0.8]);
    hold off;
    axis equal; xlim([xv(1) xv(end)]); ylim([yv(1) yv(end)]);
    title(sprintf('t = %.4f', t_val), 'FontSize', 12);
    xlabel('X'); ylabel('Y'); grid on;
end
sgtitle('Cross Section at z = mid (G = 0 contour)', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  Figure 3: Initial vs Final overlay
%  ========================================================================
if n_snap >= 2
    figure('Name', 'Initial vs Final', 'Position', [200 150 800 600], 'Color', 'w');

    [G0, ~, ~, ~, ~, t0, xv, yv, zv] = read_field(all_files{1});
    [Gf, ~, ~, ~, ~, tf, ~, ~, ~]    = read_field(all_files{end});
    [X, Y, Z] = meshgrid(xv, yv, zv);
    G0p = permute(G0, [2 1 3]);
    Gfp = permute(Gf, [2 1 3]);

    hold on;
    p1 = patch(isosurface(X, Y, Z, G0p, iso_value));
    isonormals(X, Y, Z, G0p, p1);
    set(p1, 'FaceColor', [0.3 0.8 0.3], 'EdgeColor', 'none', ...
            'FaceAlpha', 0.3, 'FaceLighting', 'gouraud');

    p2 = patch(isosurface(X, Y, Z, Gfp, iso_value));
    isonormals(X, Y, Z, Gfp, p2);
    set(p2, 'FaceColor', [0.2 0.4 1.0], 'EdgeColor', 'none', ...
            'FaceAlpha', 0.5, 'FaceLighting', 'gouraud');

    light('Position', [1 1 1]);
    light('Position', [-1 -0.5 0.5], 'Color', [.3 .3 .3]);
    axis equal;
    xlim([xv(1) xv(end)]); ylim([yv(1) yv(end)]); zlim([zv(1) zv(end)]);
    view(35, 25); grid on; box on;
    xlabel('X'); ylabel('Y'); zlabel('Z');
    legend([p1 p2], sprintf('Initial (t=%.4f)', t0), sprintf('Final (t=%.4f)', tf), ...
        'Location', 'northeast', 'FontSize', 11);
    title('Initial vs Final', 'FontSize', 14);
    hold off;
end

fprintf('Done.\n');

end
