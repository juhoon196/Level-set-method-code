%% G-equation Interface Animation
clear; close all; clc;

%% 1. Configuration
output_dir = '../output';  % 데이터 경로
nx = 1601; ny = 1601;        % 내부 격자 수
nghost = 3;                % 고스트 셀 수

% 파일 목록 가져오기 및 숫자 순서 정렬
file_list = dir(fullfile(output_dir, 'G_step_*.bin'));
filenames = {file_list.name};
step_nums = cellfun(@(x) str2double(regexp(x, '\d+', 'match')), filenames);
[~, sort_idx] = sort(step_nums);
sorted_files = filenames(sort_idx);

%% 2. Coordinate Setup
x = linspace(0, 1, nx);
y = linspace(0, 1, ny);
[X, Y] = meshgrid(x, y);

%% 3. Animation Loop
figure('Color', 'w');
ax = axes;

for k = length(sorted_files):length(sorted_files)
    % 파일 경로
    filepath = fullfile(output_dir, sorted_files{k});
    
    % 데이터 읽기 (제공된 read_level_set_binary 로직 활용)
    fid = fopen(filepath, 'rb');
    if fid == -1, continue; end
    
    % 헤더 읽기 (nx, ny, nghost)
    header = fread(fid, 3, 'int32');
    nx_total = header(1) + 2*header(3);
    ny_total = header(2) + 2*header(3);
    
    % 데이터 읽기 및 Reshape
    data = fread(fid, nx_total * ny_total, 'float64');
    fclose(fid);
    
    G_full = reshape(data, [nx_total, ny_total])';
    
    % 고스트 셀 제거 및 내부 영역 추출
    G = G_full(header(3)+1:end-header(3), header(3)+1:end-header(3));
    
    % 시각화 (G = 0 인터페이스)
    cla(ax);
    contour(ax, X, Y, G, [0 0], 'b', 'LineWidth', 2);
    
    % 타이틀 및 축 설정
    title(ax, sprintf('Step: %d', step_nums(sort_idx(k))));
    axis(ax, 'equal');
    axis(ax, [0 1 0 1]);
    grid(ax, 'on');
    
    drawnow; % 실시간 갱신
end

fprintf('애니메이션 종료 (총 %d 프레임)\n', length(sorted_files));