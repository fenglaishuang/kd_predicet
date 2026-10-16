
clear;

HERE = fileparts(mfilename('fullpath'));
ROOT = 'C:\Users\77940\OneDrive\Desktop\博士毕业\外审\外审意见1\第五章加内容';
OUT  = fullfile(HERE, 'section_5_3_5_outputs');
if ~exist(OUT, 'dir')
    mkdir(OUT);
end



branches = {'Slow', 'Fast'};
branchColors = [76 120 168; 87 156 101] / 255;

layerComponent = {}; layerInterval = {}; layerPoints = []; layerProfiles = [];
layerRMSE = []; layerMAE = []; layerBias = [];
binComponent = {}; binDepth = []; binRMSE = []; binMAE = []; binBias = []; binProfiles = [];
recComponent = {}; recProfileID = []; recNGrid = []; recDepthSpan = [];
recVariation = []; recGridRMSE = []; recProfileRMSE = [];

allData = cell(2, 1);
mats = cell(2, 1);

for ib = 1:numel(branches)
    branch = branches{ib};
    csvPath = fullfile(ROOT, sprintf('reconstruct_%s_3layers_transition', lower(branch)), ...
        sprintf('all_profiles_reconstructed_%s.csv', lower(branch)));
    T = readtable(csvPath, 'VariableNamingRule', 'preserve');

    required = {'profile_id', 'depth', 'kd_true', 'kd_reconstructed'};
    if ~all(ismember(required, T.Properties.VariableNames))
        error('Missing one or more required columns in %s.', csvPath);
    end
    profileID = double(T.profile_id);
    depth = double(T.depth);
    observed = double(T.kd_true);
    predicted = double(T.kd_reconstructed);
    if size(unique([profileID depth], 'rows'), 1) ~= height(T)
        error('Duplicate profile_id/depth pairs found in %s.', csvPath);
    end
    if any(~isfinite([profileID depth observed predicted]), 'all')
        error('Non-finite values found in %s.', csvPath);
    end
    if numel(unique(profileID)) ~= 33 || height(T) ~= 3456
        error('Unexpected profile count or row count in %s.', csvPath);
    end

    err = predicted - observed;
    D = table(profileID, depth, observed, predicted, err, ...
        'VariableNames', {'profile_id','depth','observed','predicted','error'});
    allData{ib} = D;

    intervals = [0 15; 15 30; 30 50; 50 80];
    for k = 1:size(intervals, 1)
        left = intervals(k, 1); right = intervals(k, 2);
        if right < 80
            mask = depth >= left & depth < right;
        else
            mask = depth >= left & depth <= right;
        end
        e = err(mask);
        layerComponent{end+1,1} = branch;
        layerInterval{end+1,1} = sprintf('%d-%d', left, right);
        layerPoints(end+1,1) = sum(mask);
        layerProfiles(end+1,1) = numel(unique(profileID(mask)));
        layerRMSE(end+1,1) = sqrt(mean(e.^2));
        layerMAE(end+1,1) = mean(abs(e));
        layerBias(end+1,1) = mean(e);
    end

    for left = 0:5:75
        right = left + 5;
        if left < 75
            mask = depth >= left & depth < right;
        else
            mask = depth >= left & depth <= 80;
        end
        e = err(mask);
        binComponent{end+1,1} = branch;
        binDepth(end+1,1) = left + 2.5;
        binRMSE(end+1,1) = sqrt(mean(e.^2));
        binMAE(end+1,1) = mean(abs(e));
        binBias(end+1,1) = mean(e);
        binProfiles(end+1,1) = numel(unique(profileID(mask)));
    end

    depthGrid = (0.5:0.5:80.5)';
    M = nan(numel(depthGrid), 33);
    for j = 1:height(D)
        row = find(abs(depthGrid - depth(j)) < 1e-8, 1);
        col = round(profileID(j));
        if ~isempty(row) && col >= 1 && col <= 33
            M(row, col) = err(j);
        end
    end
    mats{ib} = M;

    for pid = unique(profileID(:))'
        mask = profileID == pid;
        sd = depth(mask); so = observed(mask); se = err(mask);
        [sd, order] = sort(sd);
        so = so(order); se = se(order);
        gridDepth = (16:2:80)';
        gridDepth = gridDepth(gridDepth >= min(sd) & gridDepth <= max(sd));
        if isempty(gridDepth)
            continue;
        end
        oi = interp1(sd, so, gridDepth, 'linear');
        pi = interp1(sd, predicted(mask), gridDepth, 'linear');
        if numel(gridDepth) > 1
            span = gridDepth(end) - gridDepth(1);
        else
            span = NaN;
        end
        if numel(gridDepth) > 2
            roughness = mean(abs(gradient(oi, gridDepth)));
        else
            roughness = NaN;
        end

        recComponent{end+1,1} = branch;
        recProfileID(end+1,1) = pid;
        recNGrid(end+1,1) = numel(gridDepth);
        recDepthSpan(end+1,1) = span;
        recVariation(end+1,1) = roughness;
        recGridRMSE(end+1,1) = sqrt(mean((pi - oi).^2));
        recProfileRMSE(end+1,1) = sqrt(mean(se.^2));
    end
end

profileMetrics = table(recComponent, recProfileID, recNGrid, recDepthSpan, ...
    recVariation, recGridRMSE, recProfileRMSE, ...
    'VariableNames', {'component','profile_id','n_grid','depth_span', ...
    'vertical_variation','grid_RMSE','profile_RMSE'});
depthMetrics = table(binComponent, binDepth, binRMSE, binMAE, binBias, binProfiles, ...
    'VariableNames', {'component','depth','RMSE','MAE','Bias','profiles'});
intervalMetrics = table(layerComponent, layerInterval, layerPoints, layerProfiles, ...
    layerRMSE, layerMAE, layerBias, ...
    'VariableNames', {'component','depth_interval','points','profiles','RMSE','MAE','Bias'});

writetable(profileMetrics, fullfile(OUT, 'profile_structure_metrics.csv'));
writetable(depthMetrics, fullfile(OUT, 'depth_5m_metrics.csv'));
writetable(intervalMetrics, fullfile(OUT, 'depth_interval_metrics.csv'));

statsComponent = {}; statsRho = []; statsP = []; statsN = [];
for ib = 1:numel(branches)
    branch = branches{ib};
    s = profileMetrics(strcmp(profileMetrics.component, branch), :);
    [rho, pValue] = corr(s.vertical_variation, s.grid_RMSE, 'Type', 'Spearman', 'Rows', 'complete');
    statsComponent{end+1,1} = branch;
    statsRho(end+1,1) = rho;
    statsP(end+1,1) = pValue;
    statsN(end+1,1) = height(s);
end
statsTable = table(statsComponent, statsRho, statsP, statsN, 'VariableNames', {'component','spearman_rho','p_value','n'});
writetable(statsTable, fullfile(OUT, 'structure_associations.csv'));
