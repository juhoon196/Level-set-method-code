%% visualize_3d.m
%  G-equation Level-Set 3D solver output visualization
%
%  Reads binary output files directly (time is stored in the file header).
%  Use the slider or arrow keys to navigate through time snapshots.
%  Press Space to play/pause animation.
%
%  Binary format per file:
%    int32[4]  : nx, ny, nz, nghost
%    double[4] : time, dx, dy, dz
%    double[]  : G field data (nx_total * ny_total * nz_total)

clear; clc; close all;

%% ========================================================================
%  Configuration
%  ========================================================================
output_dir = '../output';
if ~exist(output_dir, 'dir'), output_dir = './output'; end
if ~exist(output_dir, 'dir')
    error('Output directory not found. Adjust output_dir in the script.');
end

iso_value  = 0.0;             % G = 0 is the interface
face_alpha = 0.6;
face_color = [0.2 0.6 1.0];

%% ========================================================================
%  Scan and sort binary files
%  ========================================================================
files_init  = dir(fullfile(output_dir, 'G_initial.bin'));
files_step  = dir(fullfile(output_dir, 'G_step_*.bin'));
files_final = dir(fullfile(output_dir, 'G_final.bin'));

% Build file list in time order
all_files = {};
if ~isempty(files_init)
    all_files{end+1} = fullfile(output_dir, files_init(1).name);
end
% Sort step files by step number
if ~isempty(files_step)
    step_nums = zeros(length(files_step), 1);
    for k = 1:length(files_step)
        tok = regexp(files_step(k).name, 'G_step_(\d+)\.bin', 'tokens');
        if ~isempty(tok)
            step_nums(k) = str2double(tok{1}{1});
        end
    end
    [~, sort_idx] = sort(step_nums);
    for k = 1:length(sort_idx)
        all_files{end+1} = fullfile(output_dir, files_step(sort_idx(k)).name); %#ok<SAGROW>
    end
end
if ~isempty(files_final)
    all_files{end+1} = fullfile(output_dir, files_final(1).name);
end

n_frames = length(all_files);
if n_frames == 0
    error('No binary files found in %s', output_dir);
end
fprintf('Found %d snapshots.\n', n_frames);

%% ========================================================================
%  Read header only (for quick scan of times)
%  ========================================================================
times = zeros(n_frames, 1);
for k = 1:n_frames
    fid = fopen(all_files{k}, 'rb');
    fread(fid, 4, 'int32');          % skip int header
    hd = fread(fid, 4, 'float64');   % [time, dx, dy, dz]
    times(k) = hd(1);
    fclose(fid);
end
fprintf('Time range: [%.6f, %.6f]\n\n', times(1), times(end));

%% ========================================================================
%  Read full field from binary file
%  ========================================================================
function [G_int, nx, ny, nz, nghost, t, xv, yv, zv] = read_field_bin(filepath)
    fid = fopen(filepath, 'rb');
    hi = fread(fid, 4, 'int32');
    nx = hi(1); ny = hi(2); nz = hi(3); nghost = hi(4);
    hd = fread(fid, 4, 'float64');
    t = hd(1); dx = hd(2); dy = hd(3); dz = hd(4);

    nx_t = nx + 2*nghost;
    ny_t = ny + 2*nghost;
    nz_t = nz + 2*nghost;
    data = fread(fid, nx_t*ny_t*nz_t, 'float64');
    fclose(fid);

    G_full = reshape(data, [nx_t, ny_t, nz_t]);
    G_int  = G_full(nghost+1:nghost+nx, nghost+1:nghost+ny, nghost+1:nghost+nz);

    xv = (0:nx-1) * dx;
    yv = (0:ny-1) * dy;
    zv = (0:nz-1) * dz;
end

%% ========================================================================
%  Create figure
%  ========================================================================
fig = figure('Name', 'Level-Set 3D Viewer', ...
    'NumberTitle', 'off', ...
    'Position', [100 100 1000 800], ...
    'Color', 'w', ...
    'KeyPressFcn', @keypress_cb);

ax = axes('Parent', fig, 'Position', [0.08 0.15 0.84 0.78]);

% --- Slider ---
sp = uipanel('Parent', fig, 'Position', [0.05 0.02 0.90 0.08], ...
    'BorderType', 'none', 'BackgroundColor', 'w');

if n_frames > 1
    sl = uicontrol('Parent', sp, 'Style', 'slider', ...
        'Units', 'normalized', 'Position', [0.10 0.40 0.80 0.45], ...
        'Min', 1, 'Max', n_frames, 'Value', 1, ...
        'SliderStep', [1/(n_frames-1), max(10/(n_frames-1),1/(n_frames-1))], ...
        'Callback', @slider_cb);
else
    sl = uicontrol('Parent', sp, 'Style', 'slider', ...
        'Units', 'normalized', 'Position', [0.10 0.40 0.80 0.45], ...
        'Min', 1, 'Max', 1.001, 'Value', 1, 'Enable', 'off');
end

info_lbl = uicontrol('Parent', sp, 'Style', 'text', ...
    'Units', 'normalized', 'Position', [0 0 1 0.38], ...
    'String', '', 'FontSize', 11, 'BackgroundColor', 'w', ...
    'HorizontalAlignment', 'center');

% State
state.frame = 1;
state.playing = false;
fig.UserData = state;

% Draw first frame
draw_frame(1);

%% ========================================================================
%  Drawing
%  ========================================================================
function draw_frame(fi)
    fi = max(1, min(n_frames, round(fi)));

    [G, nx, ny, nz, ~, t_val, xv, yv, zv] = read_field_bin(all_files{fi});
    [X, Y, Z] = meshgrid(xv, yv, zv);
    Gp = permute(G, [2 1 3]);   % (x,y,z) -> (y,x,z) for meshgrid

    cla(ax); hold(ax, 'on');

    p = patch(ax, isosurface(X, Y, Z, Gp, iso_value));
    isonormals(X, Y, Z, Gp, p);
    set(p, 'FaceColor', face_color, 'EdgeColor', 'none', ...
           'FaceAlpha', face_alpha, 'FaceLighting', 'gouraud', ...
           'AmbientStrength', 0.4, 'DiffuseStrength', 0.7, ...
           'SpecularStrength', 0.3);

    cap = patch(ax, isocaps(X, Y, Z, Gp, iso_value, 'below'));
    set(cap, 'FaceColor', face_color*0.8, 'EdgeColor', 'none', ...
             'FaceAlpha', face_alpha*0.8);

    light(ax, 'Position', [1 1 1], 'Style', 'infinite');
    light(ax, 'Position', [-1 -0.5 0.5], 'Style', 'infinite', 'Color', [.3 .3 .3]);

    axis(ax, 'equal');
    xlim(ax, [xv(1) xv(end)]); ylim(ax, [yv(1) yv(end)]); zlim(ax, [zv(1) zv(end)]);
    xlabel(ax, 'X'); ylabel(ax, 'Y'); zlabel(ax, 'Z');
    view(ax, 35, 25); grid(ax, 'on'); box(ax, 'on');
    hold(ax, 'off');

    title(ax, sprintf('G = 0 Isosurface   |   t = %.6f', t_val), 'FontSize', 14);
    info_lbl.String = sprintf('Frame %d / %d   |   t = %.6f', fi, n_frames, t_val);

    sl.Value = fi;
    s = fig.UserData; s.frame = fi; fig.UserData = s;
    drawnow;
end

%% ========================================================================
%  Callbacks
%  ========================================================================
function slider_cb(src, ~)
    draw_frame(round(src.Value));
end

function keypress_cb(~, evt)
    s = fig.UserData;
    switch evt.Key
        case 'rightarrow', draw_frame(s.frame + 1);
        case 'leftarrow',  draw_frame(s.frame - 1);
        case 'home',       draw_frame(1);
        case 'end',        draw_frame(n_frames);
        case 'space',      toggle_play();
        case 'escape',     stop_play();
    end
end

function toggle_play()
    s = fig.UserData;
    if s.playing, stop_play(); return; end
    s.playing = true; fig.UserData = s;
    for f = s.frame+1 : n_frames
        s2 = fig.UserData;
        if ~s2.playing, break; end
        draw_frame(f);
        pause(0.05);
    end
    stop_play();
end

function stop_play()
    s = fig.UserData; s.playing = false; fig.UserData = s;
end

%% ========================================================================
fprintf('\n=== Controls ===\n');
fprintf('  Left/Right arrow : prev/next frame\n');
fprintf('  Home/End         : first/last frame\n');
fprintf('  Space            : play/pause\n');
fprintf('  Escape           : stop\n');
fprintf('================\n\n');

end
