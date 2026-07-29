%% ============================================================
%  Kim & Forger Detailed Model — CSV export (T, R only)
%  R = min-max normalization of ode15s solution (no smoothing)
% ============================================================

clear; clc;

% ---------- Simulation ----------
dt_nom = 0.01;
tmax   = 480;          % 後半50% ≈ 10周期
tspan  = 0:dt_nom:tmax;

fprintf('Running DetailedModel (Kim & Forger 2012)...\n');

x0 = DetailedModel();
p  = DetailedModel('parametervalues');

opts = odeset('RelTol',1e-6,'AbsTol',1e-9);
[t, X] = ode15s(@(t,x) DetailedModel(t,x,p), tspan, x0, opts);

name = DetailedModel('states');

% ---------- Variables to export ----------
var_names = name;  % export all 180 states

% Resolve case-insensitive filename collisions (e.g. 'GR' vs 'Gr' -- macOS's
% default APFS volume is case-insensitive, so writing both would silently
% overwrite the same file). The state name used to look up the ODE column
% (vname) is untouched; only the output filename is disambiguated.
lower_names = lower(var_names);
file_names  = var_names;
for vi = 1:length(var_names)
    dup_idx = find(strcmp(lower_names, lower_names{vi}));
    if length(dup_idx) > 1
        rank = find(dup_idx == vi);
        file_names{vi} = sprintf('%s_case%d', var_names{vi}, rank);
    end
end

save_dir = fileparts(mfilename('fullpath'));
N_t      = size(X, 1);

% ---------- Process each variable ----------
for vi = 1:length(var_names)
    vname = var_names{vi};
    fname = file_names{vi};
    idx_v = find(strcmp(name, vname));
    if isempty(idx_v)
        warning('Variable "%s" not found, skipping.', vname);
        continue;
    end

    % --- 後半50%を抽出（≈10周期）---
    i0  = floor(0.5 * N_t);
    T_s = t(i0:end) - t(i0);
    x_s = X(i0:end, idx_v);

    % --- CSV 保存 ---
    csv_data = table(T_s(:), x_s(:), 'VariableNames', {'T', 'R'});
    writetable(csv_data, fullfile(save_dir, sprintf('KF_%s.csv', fname)));
    fprintf('Saved: KF_%s.csv\n', fname);
end

fprintf('All done.\n');
