clear; clc; close all;


%%
nx = 800; % 반드시 C++에서 설정한 NX, NY와 일치
ny = 800;
Lx = 0.02;
Ly = 0.02;

dx = Lx / (nx-1);
dy = Ly / (ny-1);

% 실제 물리 좌표 생성
x = 0:dx:Lx;
y = 0:dy:Ly;
[X, Y] = meshgrid(x, y);



for k = 5:5:255
    filename = './results/5Hz/itr_G_field_' + string(k) + '.bin';
    fid = fopen(filename, 'rb');
    Gvec = fread(fid, [nx ny], 'double');
    
    G = Gvec';
    
    

    % clf;
    
    contour(X, Y, G, [0.0 0.0], '-r', 'LineWidth', 3);
    % hold on;
    % surf(X, Y, G, 'EdgeColor', 'none')
    % colormap(jet)
    % view(2)
    % colorbar;

    xlim([0 0.02])
    ylim([0 0.02])
    axis equal;
    title(sprintf('Step %d', k));
    xlabel('x'); ylabel('y');
    grid on;
    pause(0.01);
    % hold off;
end

hold on;
% 
% for k = 3900:3900
%     filename = './results/G_result_' + string(k) + '.bin';
%     fid = fopen(filename, 'rb');
%     Gvec = fread(fid, nx*ny, 'double');
%     fclose(fid);
% 
%     G = reshape(Gvec, [nx, ny])';
% 
%     % clf;
% 
%     contour(X, Y, G, [0.0 0.0], '--r', 'LineWidth', 2);
%     % hold on;
%     % surf(X, Y, G, 'EdgeColor', 'none')
%     % colormap(jet)
%     % view(2)
%     % colorbar;
% 
%     axis([0 1 0 1]);    % 실제 도메인에 맞게 고정
%     axis equal;
%     title(sprintf('Step %d', k));
%     xlabel('x'); ylabel('y');
%     grid on;
%     pause(0.01);
%     % hold off;
% end