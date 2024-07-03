% Define the directories where the CSV files are located
dataDirs = {
     '/home/tianhu/AMP_for_hardware/Data/Uniform_6'
     % , ...
    '/home/tianhu/AMP_for_hardware/Data/Obstacle_3' ...
    % Add more directories if needed
};

% List of joints and states to plot
joints = {'left_thigh', 'right_thigh'};
states = {'dof_pos', 'dof_vel', 'dof_torque', 'base_vel_x', 'base_vel_y', 'base_vel_yaw', 'contact_forces_z', 'power', 'CoT'};
terrainLabels = {'Uniform', 'Discrete obscacles'}; % Add corresponding labels for the terrains
% terrainLabels = {'Uniform'}; % Add corresponding labels for the terrains


% Ensure each terrain has a corresponding label
if length(dataDirs) ~= length(terrainLabels)
    error('Each data directory must have a corresponding terrain label.');
end

% Initialize the time vector
timeFile = fullfile(dataDirs{1}, 'time.csv');
if isfile(timeFile)
    time = readmatrix(timeFile);
else
    error('Time file not found in the first directory.');
end

% Ensure time is a column vector
if size(time, 2) > 1
    time = time';
end

% Define the time limit (4 seconds)
timeLimit = 5;

% Find the indices of the time vector within the first 4 seconds
timeIndices = find(time <= timeLimit);

% Font and legend properties
fontSize = 20;
fontWeight = 'bold';
legendFontSize = 14;
legendWeight = 'bold';
lineWidth = 2; % Line width for plots


% Plot base velocities
baseStates = {'base_vel_x', 'base_vel_y', 'base_vel_yaw'};
baseLabels = {'Base lin X vel [m/s]', 'Base lin Y vel [m/s]', 'Base ang vel [rad/s]'};
commandLabels = {'command_x', 'command_y', 'command_yaw'};
for i = 1:length(baseStates)
    baseState = baseStates{i};
    commandLabel = commandLabels{i};
    figure;
    hold on;
    for j = 1:length(dataDirs)
        dataDir = dataDirs{j};
        terrainLabel = terrainLabels{j};
        baseFile = fullfile(dataDir, [baseState, '.csv']);
        if isfile(baseFile)
            baseVel = readmatrix(baseFile);
            if size(baseVel, 1) ~= length(time)
                error('Mismatch in time and baseVel dimensions for state %s in %s', baseState, terrainLabel);
            end
            plot(time(timeIndices), baseVel(timeIndices), 'DisplayName', [terrainLabel, ' Measured'], 'LineWidth', lineWidth);
        else
            fprintf('File not found: %s\n', baseFile);
        end
    end
    % Plot the command data only once as it is the same for all terrains
    cmdFile = fullfile(dataDirs{1}, [commandLabel, '.csv']);
    if isfile(cmdFile)
        cmd = readmatrix(cmdFile);
        if size(cmd, 1) ~= length(time)
            error('Mismatch in time and cmd dimensions for state %s', baseState);
        end
        plot(time(timeIndices), cmd(timeIndices), 'DisplayName', 'Commanded', 'LineWidth', lineWidth);
    else
        fprintf('Command file not found: %s\n', cmdFile);
    end
    xlabel('Time [s]', 'FontSize', fontSize, 'FontWeight', fontWeight);
    ylabel(baseLabels{i}, 'FontSize', fontSize, 'FontWeight', fontWeight);
    legend('Location', 'best', 'FontSize', legendFontSize, 'FontWeight', legendWeight);
    hold off;
end



% Plot contact forces
contactFile = 'contact_forces_z.csv';
% Create figure with subplots
figure('Position', [100, 100, 1200, 800]);
% Initialize y-axis limits
yLimits = [0 350];  % Set this to appropriate limits based on your data

% Subplot for Plane terrain
subplot(2, 1, 1);
hold on;
dataDir = dataDirs{1};
terrainLabel = terrainLabels{1};
contactFilePath = fullfile(dataDir, contactFile);
if isfile(contactFilePath)
    forces = readmatrix(contactFilePath);
    if size(forces, 1) > length(time)
        forces = forces(1:length(time), :);
    end
    plot(time(timeIndices), forces(timeIndices, 3), 'DisplayName', [terrainLabel, ' Left Leg'], 'LineWidth', lineWidth);  % Plot the third force
    plot(time(timeIndices), forces(timeIndices, 4), 'DisplayName', [terrainLabel, ' Right Leg'], 'LineWidth', lineWidth);  % Plot the fourth force
else
    fprintf('File not found: %s\n', contactFilePath);
end
ylabel('F_z [N]', 'FontSize', fontSize, 'FontWeight', fontWeight);
set(gca, 'FontSize', fontSize - 6, 'FontWeight', fontWeight); % Set tick label size and weight
legend('Location', 'best', 'FontSize', legendFontSize, 'FontWeight', legendWeight);
% title('Plane Terrain');
ylim(yLimits);  % Set y-axis limit
hold off;

% Subplot for Uniform terrain
dataDirs = {'/home/tianhu/AMP_for_hardware/Data/Uniform_6', '/home/tianhu/AMP_for_hardware/Data/Obstacle_3'};
terrainLabels = {'Uniform', 'Obstacle'};
subplot(2, 1, 2);
hold on;
dataDir = dataDirs{2};
terrainLabel = terrainLabels{2};
contactFilePath = fullfile(dataDir, contactFile);
if isfile(contactFilePath)
    forces = readmatrix(contactFilePath);
    if size(forces, 1) > length(time)
        forces = forces(1:length(time), :);
    end
    plot(time(timeIndices), forces(timeIndices, 3), 'DisplayName', [terrainLabel, ' Left leg'], 'LineWidth', lineWidth);  % Plot the third force
    plot(time(timeIndices), forces(timeIndices, 4), 'DisplayName', [terrainLabel, ' Right leg'], 'LineWidth', lineWidth);  % Plot the fourth force
else
    fprintf('File not found: %s\n', contactFilePath);
end
xlabel('Time [s]', 'FontSize', fontSize, 'FontWeight', fontWeight);
ylabel('F_z [N]', 'FontSize', fontSize, 'FontWeight', fontWeight);
set(gca, 'FontSize', fontSize - 6, 'FontWeight', fontWeight); % Set tick label size and weight
legend('Location', 'best', 'FontSize', legendFontSize, 'FontWeight', legendWeight);
% title('Uniform Terrain');
ylim(yLimits);  % Set y-axis limit
hold off;
% % Plot power consumption
% powerFile = 'power.csv';
% figure;
% hold on;
% for j = 1:length(dataDirs)
%     dataDir = dataDirs{j};
%     terrainLabel = terrainLabels{j};
%     powerFilePath = fullfile(dataDir, powerFile);
%     if isfile(powerFilePath)
%         power = readmatrix(powerFilePath);
%         if size(power, 1) ~= length(time)
%             error('Mismatch in time and power dimensions in %s', terrainLabel);
%         end
%         plot(time(timeIndices), power(timeIndices), 'DisplayName', [terrainLabel, ' Power Consumption'], 'LineWidth', lineWidth);
%     else
%         fprintf('File not found: %s\n', powerFilePath);
%     end
% end
% xlabel('Time [s]', 'FontSize', fontSize, 'FontWeight', fontWeight);
% ylabel('Power [W]', 'FontSize', fontSize, 'FontWeight', fontWeight);
% legend('Location', 'best', 'FontSize', legendFontSize, 'FontWeight', legendWeight);
% hold off;

% Plot Cost of Transport (CoT)
cotFile = 'CoT.csv';
figure;
hold on
for j = 1:length(dataDirs)
    dataDir = dataDirs{j};
    terrainLabel = terrainLabels{j};
    cotFilePath = fullfile(dataDir, cotFile);
    if isfile(cotFilePath)
        cot = readmatrix(cotFilePath);
        if size(cot, 1) ~= length(time)
            error('Mismatch in time and CoT dimensions in %s', terrainLabel);
        end
        plot(time(timeIndices), cot(timeIndices), 'DisplayName', [terrainLabel, ' CoT'], 'LineWidth', lineWidth);
    end
end
xlabel('Time [s]', 'FontSize', fontSize, 'FontWeight', fontWeight);
ylabel('CoT', 'FontSize', fontSize, 'FontWeight', fontWeight);
legend('Location', 'best', 'FontSize', legendFontSize, 'FontWeight', legendWeight);
hold off;

% Define colors for consistency
colors = {'b', 'r', 'k'}; % Blue for Plane, Red for Uniform, Black for Commanded

% Subplot for base velocities
baseStates = {'base_vel_x', 'base_vel_yaw'};
baseLabels = {'Base X vel [m/s]', 'Base ang vel [rad/s]'};
commandLabels = {'command_x', 'command_yaw'};
figure('Position', [100, 100, 1200, 1200]); % Adjust the figure size

ax1 = subplot(2, 1, 1);
hold on;
h = []; % Array to store plot handles for the legend
for j = 1:length(dataDirs)
    dataDir = dataDirs{j};
    terrainLabel = terrainLabels{j};
    baseFile = fullfile(dataDir, [baseStates{1}, '.csv']);
    if isfile(baseFile)
        baseVel = readmatrix(baseFile);
        if size(baseVel, 1) ~= length(time)
            error('Mismatch in time and baseVel dimensions for state %s in %s', baseStates{1}, terrainLabel);
        end
        h(j) = plot(time(timeIndices), baseVel(timeIndices), 'LineWidth', lineWidth, 'Color', colors{j});
    else
        fprintf('File not found: %s\n', baseFile);
    end
end
cmdFile = fullfile(dataDirs{1}, [commandLabels{1}, '.csv']);
if isfile(cmdFile)
    cmd = readmatrix(cmdFile);
    if size(cmd, 1) ~= length(time)
        error('Mismatch in time and cmd dimensions for state %s', baseStates{1});
    end
    h(3) = plot(time(timeIndices), cmd(timeIndices), 'LineWidth', lineWidth, 'Color', colors{3});
else
    fprintf('Command file not found: %s\n', cmdFile);
end
ylabel(baseLabels{1}, 'FontSize', fontSize, 'FontWeight', fontWeight);
set(gca, 'FontSize', fontSize - 6, 'FontWeight', fontWeight); % Set tick label size and weight
hold off;

ax2 = subplot(2, 1, 2);
hold on;
for j = 1:length(dataDirs)
    dataDir = dataDirs{j};
    terrainLabel = terrainLabels{j};
    baseFile = fullfile(dataDir, [baseStates{2}, '.csv']);
    if isfile(baseFile)
        baseVel = readmatrix(baseFile);
        if size(baseVel, 1) ~= length(time)
            error('Mismatch in time and baseVel dimensions for state %s in %s', baseStates{2}, terrainLabel);
        end
        plot(time(timeIndices), baseVel(timeIndices), 'LineWidth', lineWidth, 'Color', colors{j});
    else
        fprintf('File not found: %s\n', baseFile);
    end
end
cmdFile = fullfile(dataDirs{1}, [commandLabels{2}, '.csv']);
if isfile(cmdFile)
    cmd = readmatrix(cmdFile);
    if size(cmd, 1) ~= length(time)
        error('Mismatch in time and cmd dimensions for state %s', baseStates{2});
    end
    plot(time(timeIndices), cmd(timeIndices), 'LineWidth', lineWidth, 'Color', colors{3});
else
    fprintf('Command file not found: %s\n', cmdFile);
end

ylabel(baseLabels{2}, 'FontSize', fontSize, 'FontWeight', fontWeight);
xlabel('Time [s]', 'FontSize', fontSize, 'FontWeight', fontWeight);
set(gca, 'FontSize', fontSize - 6, 'FontWeight', fontWeight); % Set tick label size and weight
% xlabel('Time [s]', 'FontSize', fontSize, 'FontWeight', fontWeight);
hold off;

% Create a combined legend
legend([h(1), h(2), h(3)], {'Obstacle Measured', 'Uniform Measured', 'Commanded'}, 'Location', 'northoutside', 'Orientation', 'horizontal', 'FontSize', legendFontSize, 'FontWeight', legendWeight);

% Adjust figure and axis properties for saving
set(gcf, 'Position', [100, 100, 1200, 800]);
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));

% Save the figure
print(gcf, '/home/tianhu/AMP_for_hardware/Data/velocity_plot.pdf', '-dpdf', '-fillpage');
% Plot Cost of Transport (CoT)
cotFile = 'CoT.csv';
figure;
hold on
for j = 1:length(dataDirs)
    dataDir = dataDirs{j};
    terrainLabel = terrainLabels{j};
    cotFilePath = fullfile(dataDir, cotFile);
    if isfile(cotFilePath)
        cot = readmatrix(cotFilePath);
        if size(cot, 1) ~= length(time)
            error('Mismatch in time and CoT dimensions in %s', terrainLabel);
        end
        plot(time(timeIndices), cot(timeIndices), 'DisplayName', [terrainLabel, ' CoT'], 'LineWidth', lineWidth);
    end
end
xlabel('Time [s]', 'FontSize', fontSize, 'FontWeight', fontWeight);
ylabel('CoT', 'FontSize', fontSize, 'FontWeight', fontWeight);
legend('Location', 'best', 'FontSize', legendFontSize, 'FontWeight', legendWeight);
hold off;

% dataDirPlane = '/home/tianhu/AMP_for_hardware/Data/Plane';
% dataDirUniform = '/home/tianhu/AMP_for_hardware/Data/Uniform';
% % Font and legend properties
% fontSize = 24;
% fontWeight = 'bold';
% legendFontSize = 12;
% legendWeight = 'bold';
% yLabelFontSize = 28; % Increase y-label font size
% 
% % Base velocities
% baseStates = {'base_vel_x', 'base_vel_yaw', 'CoT'};
% baseLabels = {'Base X vel [m/s]', 'Base yaw vel [rad/s]', 'CoT'};
% commandLabels = {'command_x', 'command_yaw'};
% 
% figure;
% 
% for i = 1:length(baseStates)
%     baseState = baseStates{i};
%     commandLabel = commandLabels{min(i, length(commandLabels))};  % Handle CoT case
% 
%     % Plane data
%     baseFilePlane = fullfile(dataDirPlane, [baseState, '.csv']);
%     if isfile(baseFilePlane)
%         baseVelPlane = readmatrix(baseFilePlane);
%         if size(baseVelPlane, 1) ~= length(time)
%             error('Mismatch in time and baseVelPlane dimensions for state %s', baseState);
%         end
%     else
%         error('File for Plane data not found: %s', baseFilePlane);
%     end
% 
%     % Uniform data
%     baseFileUniform = fullfile(dataDirUniform, [baseState, '.csv']);
%     if isfile(baseFileUniform)
%         baseVelUniform = readmatrix(baseFileUniform);
%         if size(baseVelUniform, 1) ~= length(time)
%             error('Mismatch in time and baseVelUniform dimensions for state %s', baseState);
%         end
%     else
%         error('File for Uniform data not found: %s', baseFileUniform);
%     end
% 
%     % Commanded data
%     cmdFile = fullfile(dataDirPlane, [commandLabel, '.csv']);  % Ensure correct file path for command
%     if isfile(cmdFile)
%         cmd = readmatrix(cmdFile);
%         if size(cmd, 1) ~= length(time)
%             error('Mismatch in time and cmd dimensions for state %s', commandLabel);
%         end
%     else
%         cmd = [];  % For CoT case where there is no command data
%     end
% 
%     % Plotting
%     subplot(3, 1, i);
%     hold on;
%     h1 = plot(time, baseVelPlane, 'DisplayName', 'Plane Measured', 'LineWidth', 2);
%     h2 = plot(time, baseVelUniform, 'DisplayName', 'Uniform Measured', 'LineWidth', 2);
%     if ~isempty(cmd)
%         h3 = plot(time, cmd, 'DisplayName', 'Commanded', 'LineWidth', 2, 'Color', 'k');
%     else
%         h3 = plot(NaN, NaN, 'Color', 'k');  % Placeholder for legend consistency
%     end
%     ylabel(baseLabels{i}, 'FontSize', yLabelFontSize, 'FontWeight', fontWeight); % Increase y-label font size
%     if i == length(baseStates)
%         xlabel('Time [s]', 'FontSize', fontSize, 'FontWeight', fontWeight);
%     end
%     if i == 1
%         legend([h1, h2, h3], {'Plane Measured', 'Uniform Measured', 'Commanded'}, 'Location', 'northoutside', 'Orientation', 'horizontal', 'FontSize', legendFontSize, 'FontWeight', legendWeight);
%     end
%     set(gca, 'FontSize', fontSize - 10, 'FontWeight', fontWeight);  % Adjust tick labels size
%     hold off;
% end
% 
% Define file paths and other parameters
% Define file paths and other parameters
contactFile = 'contact_forces_z.csv';
timeIndices = 1:length(time); % Adjust timeIndices as needed

% Define terrain labels and directories
dataDirs = {'/home/tianhu/AMP_for_hardware/Data/Plane', '/home/tianhu/AMP_for_hardware/Data/Uniform'};
terrainLabels = {'Uniform', 'Obstcales'};

% Define plot parameters
lineWidth = 1.5;
fontSize = 14;
fontWeight = 'bold';
legendFontSize = 12;
legendWeight = 'bold';

% Create figure with subplots
figure('Position', [100, 100, 1200, 800]);

% Subplot for Plane terrain
subplot(2, 1, 1);
hold on;
dataDir = dataDirs{1};
terrainLabel = terrainLabels{1};
contactFilePath = fullfile(dataDir, contactFile);
if isfile(contactFilePath)
    forces = readmatrix(contactFilePath);
    if size(forces, 1) > length(time)
        forces = forces(1:length(time), :);
    end
    plot(time(timeIndices), forces(timeIndices, 3), 'DisplayName', [terrainLabel, ' RL'], 'LineWidth', lineWidth);  % Plot the third force
    plot(time(timeIndices), forces(timeIndices, 4), 'DisplayName', [terrainLabel, ' RR'], 'LineWidth', lineWidth);  % Plot the fourth force
else
    fprintf('File not found: %s\n', contactFilePath);
end
ylabel('F\_z [N]', 'FontSize', fontSize, 'FontWeight', fontWeight);
set(gca, 'FontSize', fontSize - 6, 'FontWeight', fontWeight); % Set tick label size and weight
legend('Location', 'best', 'FontSize', legendFontSize, 'FontWeight', legendWeight);
title('Plane Terrain');
hold off;

% Subplot for Uniform terrain
subplot(2, 1, 2);
hold on;
dataDir = dataDirs{2};
terrainLabel = terrainLabels{2};
contactFilePath = fullfile(dataDir, contactFile);
if isfile(contactFilePath)
    forces = readmatrix(contactFilePath);
    if size(forces, 1) > length(time)
        forces = forces(1:length(time), :);
    end
    plot(time(timeIndices), forces(timeIndices, 3), 'DisplayName', [terrainLabel, ' RL'], 'LineWidth', lineWidth);  % Plot the third force
    plot(time(timeIndices), forces(timeIndices, 4), 'DisplayName', [terrainLabel, ' RR'], 'LineWidth', lineWidth);  % Plot the fourth force
else
    fprintf('File not found: %s\n', contactFilePath);
end
xlabel('Time [s]', 'FontSize', fontSize, 'FontWeight', fontWeight);
ylabel('F\_z [N]', 'FontSize', fontSize, 'FontWeight', fontWeight);
set(gca, 'FontSize', fontSize - 6, 'FontWeight', fontWeight); % Set tick label size and weight
legend('Location', 'best', 'FontSize', legendFontSize, 'FontWeight', legendWeight);
title('Uniform Terrain');
hold off;