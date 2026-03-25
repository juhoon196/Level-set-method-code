clear all;
close all;
clc;
%%
dx = 2.5e-5;
dt = 1e-4;
t = 1600;

load("./1D_WENO_L_110Hz_0.05.mat")
ENO_L = result;
ENO_R = fliplr(result);
ENO_R(:,1) = [];
xi = [ENO_L ENO_R];

%%
% Steady solution

x1 = -0.005:dx:0;
steady_sol1 = tan(60*pi/180)*(x1+0.005);
x2 = dx:dx:0.005;
steady_sol2 = -tan(60*pi/180)*(x2-0.005);

x = [x1 x2];
steady_sol = [steady_sol1 steady_sol2];

figure; hold on;
plot(x, steady_sol, '--k', 'LineWidth', 3)

% 1D WENO solution
plot(x, xi(t, :), 'b', 'LineWidth', 3);
title(t)


% 2D level-set solution
nx = 800; ny = 800;
lx = 0.01; ly = 0.02;
dx = 2*lx/(nx-1); dy = ly/(ny-1);
Gx = -lx:dx:lx; Gy = 0:dy:ly;

f1 = "./results/110Hz/itr_G_field_" + string(t) + ".bin";
fileID = fopen(f1);
    
Gvec = fread(fileID, [nx ny],'double');
G = Gvec';
[C, h] = contour(Gx, Gy, G, [0 0], 'r', 'LineWidth', 3);
contour_x = [];
contour_y = [];
col = 1;
while col < size(C,2)
    level = C(1,col);
    n_points = C(2,col);
    xdata = C(1, col+1 : col+n_points);
    ydata = C(2, col+1 : col+n_points);

    if level == 0
        contour_x = [contour_x, NaN, xdata];  % 여러 등고선 구간 나누기 위해 NaN 추가
        contour_y = [contour_y, NaN, ydata];
    end

    col = col + n_points + 1;
end
valid_idx = ~isnan(contour_x) & ~isnan(contour_y);
x_data = contour_x(valid_idx);
y_data = contour_y(valid_idx);


% x_data가 단조 증가하지 않으면 보간이 불안정할 수 있으므로 정렬
[x_data_sorted, sort_idx] = sort(x_data);
y_data_sorted = y_data(sort_idx);

% 중복 x 제거 (interp1은 단조 increasing x 필요)
[x_data_unique, ia] = unique(x_data_sorted, 'stable');
y_data_unique = y_data_sorted(ia);

% 선형 보간 (x1 내 범위만 보간됨, 외삽은 제외)
y_interp = interp1(x_data_unique, y_data_unique, x, 'linear', 'extrap');  % 또는 'spline'

plot(x, y_interp, '--m', 'LineWidth', 2);

xlim([-0.01 0.01])
ylim([0 0.02])
axis equal
grid on;
xlabel('x')
ylabel('Flame position')


%% Comparison


level_sol_pos = y_interp - steady_sol;
xi_sol_pos = xi(t, :) - steady_sol;

% figure;
fig = figure('Visible', 'off');  % 화면에 안 띄움

hold on;
plot(x(1:199), level_sol_pos(1:199), '-r', 'LineWidth', 3)
plot(x(1:199), xi_sol_pos(1:199), '--b', 'LineWidth', 2.5)
grid on;
xlim([-0.005 0])

saveas(gcf, './figure/psi=0_ep=0.15_mu=0.15_x_tip_on.svg'); %그림 svg파일로 저장


