%% G-equation Interface Animation 3D
clear; close all; clc;

%% 1. Configuration
output_dir = '../output';  % 데이터 경로
nx = 256; ny = 256; nz = 256; % 3D 초기 격자 설정 (참조용, 실제는 헤더에서 읽음)
nghost = 3;                % 고스트 셀 수

% 파일 목록 가져오기 및 숫자 순서 정렬
file_list = dir(fullfile(output_dir, 'G_step_*.bin'));
if isempty(file_list)
    fprintf('파일이 없습니다: %s\n', output_dir);
    return;
end

filenames = {file_list.name};
step_nums = cellfun(@(x) str2double(regexp(x, '\d+', 'match')), filenames);
[~, sort_idx] = sort(step_nums);
sorted_files = filenames(sort_idx);

%% 2. Coordinate Setup & Visualization Init
figure('Color', 'w', 'Position', [100 100 1000 800]);
ax = axes;
view(3);
axis equal;
axis([0 1 0 1 0 1]);
grid on;
hold on;
xlabel('X'); ylabel('Y'); zlabel('Z');
% light('Position', [1 1 1], 'Style', 'infinite');
% lighting gouraud;
% material shiny;

%% 3. Animation Loop
for k = 1:length(sorted_files)
    % 파일 경로
    filepath = fullfile(output_dir, sorted_files{k});
    
    % 데이터 읽기
    fid = fopen(filepath, 'rb');
    if fid == -1, continue; end
    
    % 헤더 읽기 (nx, ny, nz, nghost - 4 integers 가정)
    header = fread(fid, 4, 'int32');
    if length(header) < 4
        % 3D 형식이 아닌 경우 Skip
        fclose(fid);
        warning('Skipping %s: Header size mismatch (Expected 4 [nx,ny,nz,ng] for 3D)', sorted_files{k});
        continue;
    end
    
    nx_in = header(1);
    ny_in = header(2);
    nz_in = header(3);
    nghost_in = header(4);
    
    nx_total = nx_in + 2*nghost_in;
    ny_total = ny_in + 2*nghost_in;
    nz_total = nz_in + 2*nghost_in;
    
    % 데이터 읽기 (전체)
    data = fread(fid, nx_total * ny_total * nz_total, 'float64');
    fclose(fid);
    
    if length(data) ~= nx_total * ny_total * nz_total
        warning('Data size mismatch in %s', sorted_files{k});
        continue;
    end
    
    % Reshape (nx, ny, nz)
    % C++: usually x-fastest (i), then y (j), then z (k)
    % Matlab reshape fills columns first (dimension 1 -> nx)
    G_full = reshape(data, [nx_total, ny_total, nz_total]);
    
    % 고스트 셀 제거 및 내부 영역 추출
    G = G_full(nghost_in+1:end-nghost_in, ...
               nghost_in+1:end-nghost_in, ...
               nghost_in+1:end-nghost_in);
    
    % 좌표 생성
    x = linspace(0, 1, nx_in);
    y = linspace(0, 1, ny_in);
    z = linspace(0, 1, nz_in);
    [X, Y, Z] = meshgrid(x, y, z);
    
    % 시각화 (Isosurface G=0)
    cla(ax);
    
    % Matlab meshgrid 생성시 X는 [ny, nx, nz] 형태 (rows are y, cols are x)
    % G는 [nx, ny, nz] 형태.
    % isosurface 사용 시 차원 일치를 위해 G를 permute하여 [ny, nx, nz]로 맞춤
    G_plot = permute(G, [2 1 3]);
    
    % p = patch(ax, isosurface(X, Y, Z, G_plot, 0));
    isosurface(X, Y, Z, G_plot, 0);
    % p.FaceColor = 'red';
    % p.EdgeColor = 'none';
    % p.FaceAlpha = 0.7;
    camlight;
    % 타이틀 및 설정
    title(ax, sprintf('Step: %d (Grid: %dx%dx%d)', step_nums(sort_idx(k)), nx_in, ny_in, nz_in));
    drawnow;
    pause(0.1)
end

fprintf('애니메이션 종료 (총 %d 프레임)\n', length(sorted_files));