function main(varargin)
% MATLAB port of main.py for reproducing wireless FL experiments.

args = struct( ...
    'figure', 'all', ...
    'config', '', ...
    'plot', true, ...
    'output_dir', 'outputs', ...
    'seed', [], ...
    'verbose', false);

i = 1;
while i <= numel(varargin)
    key = lower(strrep(char(varargin{i}), '--', ''));
    if strcmp(key, 'plot') || strcmp(key, 'verbose')
        if i < numel(varargin) && (islogical(varargin{i + 1}) || isnumeric(varargin{i + 1}))
            args.(key) = logical(varargin{i + 1});
            i = i + 2;
            continue;
        end
        args.(key) = true;
        i = i + 1;
        continue;
    end
    if i == numel(varargin)
        error('Missing value for argument %s', key);
    end
    value = varargin{i + 1};
    switch strrep(key, '-', '_')
        case 'figure'
            args.figure = char(value);
        case 'config'
            args.config = char(value);
        case 'output_dir'
            args.output_dir = char(value);
        case 'seed'
            args.seed = double(value);
        otherwise
            error('Unknown argument: %s', key);
    end
    i = i + 2;
end

if exist(fullfile('docs', 'Wireless-FL'), 'dir')
    addpath(fullfile('docs', 'Wireless-FL'));
end
if ~exist(args.output_dir, 'dir')
    mkdir(args.output_dir);
end

figure_names = {'3', '4', '5', '6', '7', '8', '9', '10'};
if ~strcmp(args.figure, 'all')
    figure_names = {args.figure};
end

for i = 1:numel(figure_names)
    name = figure_names{i};
    config_path = args.config;
    if isempty(config_path)
        config_path = fullfile('configs', sprintf('figure_%s.yaml', name));
    end
    contexts = build_contexts(config_path);
    keep = false(1, numel(contexts));
    for j = 1:numel(contexts)
        keep(j) = strcmp(contexts{j}.meta.figure, name);
        contexts{j}.loss.meta.plot = args.plot;
        contexts{j}.loss.meta.verbose = args.verbose;
        contexts{j}.loss.meta.output_dir = args.output_dir;
        if ~isempty(args.seed)
            contexts{j}.seed = args.seed;
            contexts{j}.loss.seed = args.seed;
            contexts{j}.loss.meta.run_seeds = args.seed;
        end
    end
    contexts = contexts(keep);
    if isempty(contexts)
        error('%s did not define figure_%s', config_path, name);
    end

    switch name
        case '3'
            result = figure_3(contexts);
        case '4'
            result = figure_4(contexts);
        case '5'
            result = figure_5(contexts);
        case '6'
            result = figure_6(contexts);
        case '7'
            result = figure_7(contexts);
        case '8'
            result = figure_8(contexts);
        case '9'
            result = figure_9(contexts);
        case '10'
            result = figure_10(contexts);
        otherwise
            error('Unsupported figure: %s', name);
    end

    if isfield(result, 'figure') && isgraphics(result.figure)
        save_dir = fullfile(args.output_dir, sprintf('figure_%s', name));
        if ~exist(save_dir, 'dir')
            mkdir(save_dir);
        end
        format_figure(result.figure, name);
        save_path = fullfile(save_dir, sprintf('001_contexts_%d.png', numel(contexts)));
        print(result.figure, save_path, '-dpng', '-r200');
        close(result.figure);
        fprintf('saved: %s\n', save_path);
    end
    print_result_summary(name, result);
    fprintf('figure_%s complete\n', name);
end

    function format_figure(fig, name)
        axs = findall(fig, 'Type', 'axes');
        for ax = reshape(axs, 1, [])
            if ~strcmp(name, '10')
                axes(ax); %#ok<LAXES>
                grid on;
                set(ax, 'gridcolor', [0.85, 0.85, 0.85]);
                set(ax, 'tickdir', 'in');
                set(ax, 'box', 'on');
                set(ax, 'linewidth', 1.0);
            end
    end
end

function print_result_summary(name, result)
switch name
    case '3'
        fprintf('figure_3 summary: proposed_loss=%.6g baseline_a_loss=%.6g baseline_b_loss=%.6g\n', ...
            result.proposed_loss, result.baseline_a_loss, result.baseline_b_loss);
    case '4'
        fprintf('figure_4 samples_per_user=%s\n', format_vector(result.samples_per_user));
        fprintf('figure_4 proposed_loss=%s\n', format_vector(result.loss.proposed));
        fprintf('figure_4 baseline_a_loss=%s\n', format_vector(result.loss.baseline_a));
        fprintf('figure_4 baseline_b_loss=%s\n', format_vector(result.loss.baseline_b));
    case '5'
        fprintf('figure_5 users=%s\n', format_vector(result.users));
        names = fieldnames(result.edge_weight_evaluations);
        for ii = 1:numel(names)
            fprintf('figure_5 %s_iterations=%s\n', names{ii}, format_vector(result.edge_weight_evaluations.(names{ii})));
        end
    case '6'
        fprintf('figure_6 users=%s\n', format_vector(result.users));
        fprintf('figure_6 theoretical_gap=%s\n', format_vector(result.theoretical_gap));
        fprintf('figure_6 simulation_gap=%s\n', format_vector(result.simulation_gap));
    case '7'
        names = fieldnames(result.accuracy);
        for ii = 1:numel(names)
            values = mean(result.accuracy.(names{ii}), 1);
            fprintf('figure_7 %s_final_accuracy=%.6g\n', names{ii}, values(end));
        end
    case '8'
        fprintf('figure_8 users=%s\n', format_vector(result.users));
        names = fieldnames(result.accuracy);
        for ii = 1:numel(names)
            fprintf('figure_8 %s_accuracy=%s\n', names{ii}, format_vector(result.accuracy.(names{ii})));
        end
    case '9'
        fprintf('figure_9 rbs=%s\n', format_vector(result.rbs));
        names = fieldnames(result.accuracy);
        for ii = 1:numel(names)
            fprintf('figure_9 %s_accuracy=%s\n', names{ii}, format_vector(result.accuracy.(names{ii})));
        end
    case '10'
        fprintf('figure_10 proposed_accuracy=%.6g baseline_b_accuracy=%.6g proposed_correct_%d=%d baseline_b_correct_%d=%d\n', ...
            result.proposed_accuracy, result.baseline_b_accuracy, result.display_count, result.proposed_correct, ...
            result.display_count, result.baseline_b_correct);
end
end

function text = format_vector(values)
if iscell(values)
    values = cell2mat(values);
end
values = double(values(:)).';
parts = cell(1, numel(values));
for ii = 1:numel(values)
    parts{ii} = sprintf('%.6g', values(ii));
end
text = ['[' strjoin(parts, ', ') ']'];
end
end

function ctx = Context(varargin)
fields = { ...
    'seed', 'task', 'partitions', 'test_data', 'counts', 'num_users', ...
    'model_factory', 'initial_model_state', 'model_bits', 'num_rbs', ...
    'distances', 'channel_gain', 'downlink_gain', 'interference', ...
    'downlink_interference', 'powers', 'packet_errors', 'uplink_rates', ...
    'downlink_rates', 'total_delays', 'energies', 'feasible', 'delay_s', ...
    'energy_j', 'pmax_w', 'rounds', 'local_epochs', 'learning_rate', ...
    'batch_size', 'device', 'loss', 'meta'};
ctx = struct();
for i = 1:numel(fields)
    ctx.(fields{i}) = [];
end
for i = 1:2:numel(varargin)
    ctx.(char(varargin{i})) = varargin{i + 1};
end
end

function data = generate_synthetic_data(varargin)
p = inputParser;
addParameter(p, 'num_users', 15);
addParameter(p, 'samples_per_user', 30);
addParameter(p, 'seed', 42);
addParameter(p, 'noise_std', 0.4);
parse(p, varargin{:});

rng(p.Results.seed, 'twister');
counts = p.Results.samples_per_user;
if isscalar(counts)
    counts = repmat(double(counts), 1, p.Results.num_users);
else
    counts = double(counts(:)).';
end
users = cell(1, numel(counts));
all_x = [];
all_y = [];
for i = 1:numel(counts)
    x = rand(counts(i), 1);
    y = -2.0 .* x + 1.0 + p.Results.noise_std .* randn(counts(i), 1);
    users{i} = struct('X', x, 'Y', y);
    all_x = [all_x; x]; %#ok<AGROW>
    all_y = [all_y; y]; %#ok<AGROW>
end
data = struct('users', {users}, 'x', all_x, 'y', all_y, ...
    'counts', counts, 'task', 'regression');
end

function data = load_mnist_data(varargin)
p = inputParser;
addParameter(p, 'num_users', 15);
addParameter(p, 'samples_per_user', 200);
addParameter(p, 'test_samples', 1000);
addParameter(p, 'seed', 42);
addParameter(p, 'data_dir', '');
addParameter(p, 'download', false);
addParameter(p, 'train_order', 'shuffled');
addParameter(p, 'partition_order', 'shuffled');
parse(p, varargin{:});

paths = {};
if ~isempty(p.Results.data_dir)
    paths{end + 1} = p.Results.data_dir; %#ok<AGROW>
end
paths = [paths, {'mnist.mat', fullfile('data', 'mnist.mat'), '/home/imes-server6/dataset/mnist.mat', ...
    'mnist.npz', fullfile('data', 'mnist.npz'), '/home/imes-server6/dataset/mnist.npz'}];

loaded = false;
for i = 1:numel(paths)
    if exist(paths{i}, 'file')
        if ends_with(paths{i}, '.mat')
            [x_train, y_train, x_test, y_test] = load_mat_mnist(paths{i});
        else
            [x_train, y_train, x_test, y_test] = load_npz_mnist(paths{i});
        end
        loaded = true;
        break;
    end
end

if ~loaded
    train_images = 'train-images-idx3-ubyte';
    train_labels = 'train-labels-idx1-ubyte';
    test_images = 't10k-images-idx3-ubyte';
    test_labels = 't10k-labels-idx1-ubyte';
    if exist(train_images, 'file') && exist(train_labels, 'file') && exist(test_images, 'file') && exist(test_labels, 'file')
        [tr_img, tr_lbl] = mnist_parse(train_images, train_labels);
        [te_img, te_lbl] = mnist_parse(test_images, test_labels);
        x_train = double(permute(tr_img, [3, 1, 2])) ./ 255.0;
        y_train = double(tr_lbl(:)) + 1;
        x_test = double(permute(te_img, [3, 1, 2])) ./ 255.0;
        y_test = double(te_lbl(:)) + 1;
        loaded = true;
    end
end

if ~loaded
    error('MNIST data not found. Provide mnist.npz or IDX files before running MNIST figures.');
end

rng(p.Results.seed, 'twister');
counts = p.Results.samples_per_user;
if isscalar(counts)
    counts = repmat(double(counts), 1, p.Results.num_users);
else
    counts = double(counts(:)).';
    if numel(counts) < p.Results.num_users
        counts = repmat(counts, 1, ceil(p.Results.num_users / numel(counts)));
    end
    counts = counts(1:p.Results.num_users);
end

total_train = min(size(x_train, 1), sum(counts));
switch lower(p.Results.train_order)
    case 'shuffled'
        train_idx = randperm(size(x_train, 1), total_train);
    case 'file_order'
        train_idx = 1:total_train;
    otherwise
        error('train_order must be shuffled or file_order');
end
test_count = min(size(x_test, 1), p.Results.test_samples);
test_idx = randperm(size(x_test, 1), test_count);

train = struct('X', flatten_mnist(x_train(train_idx, :, :)), ...
    'X4', to_mnist4d(x_train(train_idx, :, :)), ...
    'Y', y_train(train_idx));
test = struct('X', flatten_mnist(x_test(test_idx, :, :)), ...
    'X4', to_mnist4d(x_test(test_idx, :, :)), ...
    'Y', y_test(test_idx));

users = get_partitioned_data(train, 'num_users', p.Results.num_users, ...
    'samples_per_user', counts, 'seed', p.Results.seed, ...
    'non_iid', strcmpi(p.Results.partition_order, 'label_sorted'), ...
    'shuffle', strcmpi(p.Results.partition_order, 'shuffled'));
data = struct('train', train, 'test', test, 'users', {users}, 'task', 'mnist');
end

function users = get_partitioned_data(data, varargin)
p = inputParser;
addParameter(p, 'num_users', 15);
addParameter(p, 'samples_per_user', []);
addParameter(p, 'seed', 42);
addParameter(p, 'non_iid', false);
addParameter(p, 'shuffle', true);
parse(p, varargin{:});

if nargin == 0 || isempty(data)
    generated = generate_synthetic_data('num_users', p.Results.num_users, ...
        'samples_per_user', p.Results.samples_per_user, 'seed', p.Results.seed);
    users = generated.users;
    return;
end

rng(p.Results.seed, 'twister');
n = size(data.X, 1);
counts = p.Results.samples_per_user;
if isempty(counts)
    counts = repmat(floor(n / p.Results.num_users), 1, p.Results.num_users);
elseif isscalar(counts)
    counts = repmat(double(counts), 1, p.Results.num_users);
else
    counts = double(counts(:)).';
end
counts = max(0, min(counts, n));

if p.Results.non_iid
    [~, order] = sort(data.Y);
elseif p.Results.shuffle
    order = randperm(n);
else
    order = 1:n;
end

users = cell(1, numel(counts));
cursor = 1;
for i = 1:numel(counts)
    selected = order(cursor:min(cursor + counts(i) - 1, n));
    cursor = cursor + counts(i);
    if isempty(selected)
        selected = order(1);
    end
    user = struct('X', data.X(selected, :), 'Y', data.Y(selected, :));
    if isfield(data, 'X4')
        user.X4 = data.X4(:, :, :, selected);
    end
    users{i} = user;
end
end

function model = RegressionFNN(varargin)
p = inputParser;
addParameter(p, 'hidden_size', 20);
addParameter(p, 'activation', 'tanh');
parse(p, varargin{:});
h = p.Results.hidden_size;
model = struct();
model.type = 'regression_fnn';
model.activation = lower(p.Results.activation);
model.params.W1 = 0.1 .* randn(h, 1);
model.params.b1 = zeros(h, 1);
model.params.W2 = 0.1 .* randn(1, h);
model.params.b2 = 0;
end

function model = MNISTFNN(varargin)
p = inputParser;
addParameter(p, 'hidden_size', 50);
addParameter(p, 'activation', 'relu');
parse(p, varargin{:});
h = p.Results.hidden_size;
model = struct();
model.type = 'mnist_fnn';
model.activation = lower(p.Results.activation);
model.params.W1 = 0.05 .* randn(h, 28 * 28);
model.params.b1 = zeros(h, 1);
model.params.W2 = 0.05 .* randn(10, h);
model.params.b2 = zeros(10, 1);
end

function model = MNISTCNN()
if exist('dlnetwork', 'class') == 8 || exist('dlnetwork', 'file') == 2
    layers = [
        imageInputLayer([28, 28, 1], 'Normalization', 'none', 'Name', 'input')
        convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
        fullyConnectedLayer(50, 'Name', 'fc1')
        reluLayer('Name', 'relu3')
        fullyConnectedLayer(10, 'Name', 'fc2')];
    model = struct('type', 'mnist_dlnetwork', 'net', dlnetwork(layerGraph(layers)));
else
    model = MNISTFNN('hidden_size', 50, 'activation', 'relu');
    model.type = 'mnist_cnn_fallback';
end
end

function [global_model, successes] = fl_one_round(context, global_model, selected_users, assigned_rbs, selected_errors, varargin)
p = inputParser;
addParameter(p, 'round_index', 0);
addParameter(p, 'aggregation', 'fedavg');
addParameter(p, 'verbose', false);
addParameter(p, 'scheme', 'fl');
parse(p, varargin{:});

successes = 0;
local_models = {};
local_weights = [];
train_cfg = context.loss.training;
lr = context.learning_rate;
if isempty(lr)
    if strcmp(context.task, 'regression')
        lr = train_cfg.regression_lr;
    else
        lr = train_cfg.mnist_lr;
    end
end
force_first = isfield(train_cfg, 'force_first_round_success') && train_cfg.force_first_round_success;

for k = 1:numel(selected_users)
    user_idx = selected_users(k);
    local = global_model;
    user_data = context.partitions{user_idx};
    for epoch = 1:context.local_epochs %#ok<NASGU>
        local = train_full_batch(local, user_data, context.task, lr, train_cfg);
    end
    packet_error = selected_errors(k);
    accepted = (force_first && p.Results.round_index == 0 && packet_error < 1.0) || (rand() > packet_error);
    if accepted
        successes = successes + 1;
        local_models{end + 1} = local; %#ok<AGROW>
        if strcmpi(p.Results.aggregation, 'uniform')
            local_weights(end + 1) = 1.0; %#ok<AGROW>
        else
            local_weights(end + 1) = size(user_data.X, 1); %#ok<AGROW>
        end
    end
end

if ~isempty(local_models)
    global_model = average_models(local_models, local_weights);
end
end

function output = baseline_a(varargin)
output = proposed_algorithm(varargin{:}, 'scheme', 'baseline_a');
end

function output = baseline_b(varargin)
output = proposed_algorithm(varargin{:}, 'scheme', 'baseline_b');
end

function output = baseline_c(varargin)
output = proposed_algorithm(varargin{:}, 'scheme', 'baseline_c');
end

function output = proposed_algorithm(varargin)
[partitions, model, opts] = parse_algorithm_inputs(varargin{:});
cfg = opts.config;
wireless = cfg.wireless;
train_cfg = cfg.training;
seed = opts.seed;
rng(seed, 'twister');

if isempty(partitions)
    counts = zeros(1, opts.num_users);
    cycle = wireless.ki_cycle(:).';
    for i = 1:opts.num_users
        counts(i) = cycle(mod(i - 1, numel(cycle)) + 1);
    end
else
    opts.num_users = numel(partitions);
    counts = zeros(1, opts.num_users);
    for i = 1:opts.num_users
        counts(i) = size(partitions{i}.X, 1);
    end
end

if isempty(model)
    if strcmp(opts.task, 'regression')
        model = RegressionFNN();
    else
        model = MNISTFNN('activation', train_cfg.mnist_activation);
    end
elseif isa(model, 'function_handle')
    if strcmp(func2str(model), 'MNISTFNN')
        model = model('activation', train_cfg.mnist_activation);
    else
        model = model();
    end
end

if isempty(opts.model_bits)
    opts.model_bits = count_model_parameters(model) * double(train_cfg.quantization_bits);
end

link = compute_wireless_links(wireless, opts.num_users, opts.num_rbs, opts.model_bits, seed);
[allocation, selected_users, assigned_rbs, solver_iterations] = solve_assignment(opts.scheme, link, counts);

selected_errors = ones(1, numel(selected_users));
selected_powers = zeros(1, numel(selected_users));
for k = 1:numel(selected_users)
    u = selected_users(k);
    r = assigned_rbs(k);
    if link.feasible(u, r)
        selected_errors(k) = link.packet_errors(u, r);
    end
    selected_powers(k) = link.powers(u, r);
end

metrics = struct('loss', [], 'accuracy', [], 'successful_users', []);
trained_state = [];
if opts.rounds > 0 && ~isempty(partitions)
    round_context = Context( ...
        'seed', seed, 'task', opts.task, 'partitions', partitions, ...
        'test_data', opts.test_data, 'counts', counts, 'num_users', opts.num_users, ...
        'model_bits', opts.model_bits, 'num_rbs', opts.num_rbs, ...
        'powers', link.powers, 'packet_errors', link.packet_errors, ...
        'uplink_rates', link.uplink_rates, 'downlink_rates', link.downlink_rates, ...
        'total_delays', link.total_delays, 'energies', link.energies, ...
        'feasible', link.feasible, 'delay_s', wireless.delay_s, ...
        'energy_j', wireless.energy_j, 'pmax_w', wireless.pmax_w, ...
        'rounds', opts.rounds, 'local_epochs', opts.local_epochs, ...
        'learning_rate', opts.learning_rate, 'batch_size', opts.batch_size, ...
        'device', train_cfg.device, 'loss', cfg);
    global_model = model;
    aggregation = 'fedavg';
    if strcmpi(opts.scheme, 'baseline_c')
        aggregation = 'uniform';
    end
    for round_idx = 1:opts.rounds
        [global_model, successes] = fl_one_round(round_context, global_model, selected_users, assigned_rbs, selected_errors, ...
            'round_index', round_idx - 1, 'aggregation', aggregation, 'scheme', opts.scheme);
        eval_result = evaluate_model(global_model, opts.test_data, partitions, opts.task, train_cfg);
        metrics.loss(end + 1) = eval_result.loss; %#ok<AGROW>
        metrics.accuracy(end + 1) = eval_result.accuracy; %#ok<AGROW>
        metrics.successful_users(end + 1) = successes; %#ok<AGROW>
    end
    trained_state = global_model;
end

scheme = opts.scheme;
if strcmpi(scheme, 'proposed')
    scheme = 'proposed';
end
output = struct( ...
    'scheme', scheme, ...
    'allocation', allocation, ...
    'selected_users', selected_users, ...
    'assigned_rbs', assigned_rbs, ...
    'powers', link.powers, ...
    'selected_powers', selected_powers, ...
    'packet_errors', link.packet_errors, ...
    'selected_packet_errors', selected_errors, ...
    'feasible', link.feasible, ...
    'uplink_rates', link.uplink_rates, ...
    'total_delays', link.total_delays, ...
    'energies', link.energies, ...
    'model_bits', opts.model_bits, ...
    'counts', counts, ...
    'solver_iterations', solver_iterations, ...
    'metrics', metrics, ...
    'model_state', trained_state, ...
    'wireless', link);
end

function contexts = build_contexts(path)
if nargin == 0 || isempty(path)
    path = fullfile('configs', 'sweep.yaml');
end
if ~exist(path, 'file')
    error('Configuration file is required for non-paper parameters: %s', path);
end
override = parse_yaml(path);

required.wireless = {'waterfall_threshold', 'channel_model', 'interference_w', ...
    'interference_w_by_rbs', 'downlink_interference_w', 'min_distance_m', ...
    'interference_lognormal_sigma', 'rate_floor', 'channel_gain_floor', ...
    'snr_denominator_floor', 'power_floor_w', 'power_solver_maxiter', ...
    'heuristic_max_rbs'};
required.training = {'quantization_bits', 'eval_batch_size', 'mnist_activation', ...
    'mnist_train_order', 'mnist_partition_order', 'force_first_round_success', ...
    'regression_loss', 'regression_scale_floor'};
missing = {};
for section_cell = fieldnames(required).'
    section = section_cell{1};
    if ~isfield(override, section)
        for k = 1:numel(required.(section))
            missing{end + 1} = [section '.' required.(section){k}]; %#ok<AGROW>
        end
        continue;
    end
    for k = 1:numel(required.(section))
        if ~isfield(override.(section), required.(section){k})
            missing{end + 1} = [section '.' required.(section){k}]; %#ok<AGROW>
        end
    end
end
if ~isempty(missing)
    error('configs/*.yaml must explicitly provide non-paper parameters: %s', strjoin(missing, ', '));
end

config = default_config();
config = merge_structs(config, override);
config = collapse_singletons(config);
if ~isfield(config, 'figures')
    error('configs/*.yaml must explicitly provide figures');
end

run_seeds = as_vector(config.seed);
figure_fields = fieldnames(override.figures);
contexts = cell(1, numel(figure_fields));
for i = 1:numel(figure_fields)
    figure_key = figure_fields{i};
    figure_name = strrep(figure_key, 'figure_', '');
    run_config = config;
    run_config.meta = struct('figure', figure_name, 'plot', false, ...
        'run_seeds', run_seeds, 'verbose', false, 'output_dir', 'outputs');
    contexts{i} = Context('seed', run_seeds(1), 'loss', run_config, ...
        'meta', struct('figure', figure_name));
end
end

function result = figure_3(contexts)
context = contexts{1};
cfg = context.loss;
fcfg = cfg.figures.figure_3;
seed = context.seed;
data_count = first_value(fcfg.data_count);
test_count = first_value(fcfg.test_count);
rounds = first_value(fcfg.rounds);
num_rbs = first_value(fcfg.num_rbs);
local_epochs = first_value(fcfg.local_epochs);
activation = char(first_value(fcfg.activation));
learning_rate = first_value(fcfg.learning_rate);

base = floor(data_count / 15);
counts = repmat(base, 1, 15);
counts(1:mod(data_count, 15)) = counts(1:mod(data_count, 15)) + 1;
data = generate_synthetic_data('num_users', 15, 'samples_per_user', counts, 'seed', seed);
rng(seed, 'twister');
test_x = linspace(0, 1, test_count).';
test_y = -2 .* test_x + 1 + 0.4 .* randn(test_count, 1);
test_data = struct('X', test_x, 'Y', test_y);
rng(seed, 'twister');
initial = RegressionFNN('activation', activation);

schemes = {'proposed', 'optimal', 'baseline_a', 'baseline_b'};
outputs = struct();
for i = 1:numel(schemes)
    local = initial;
    switch schemes{i}
        case 'baseline_a'
            out = baseline_a(data.users, local, 'task', 'regression', 'test_data', test_data, ...
                'rounds', rounds, 'num_rbs', num_rbs, 'local_epochs', local_epochs, ...
                'learning_rate', learning_rate, 'seed', seed, 'config', cfg);
        case 'baseline_b'
            out = baseline_b(data.users, local, 'task', 'regression', 'test_data', test_data, ...
                'rounds', rounds, 'num_rbs', num_rbs, 'local_epochs', local_epochs, ...
                'learning_rate', learning_rate, 'seed', seed, 'config', cfg);
        case 'optimal'
            out = proposed_algorithm(data.users, local, 'task', 'regression', 'test_data', test_data, ...
                'rounds', rounds, 'num_rbs', num_rbs, 'local_epochs', local_epochs, ...
                'learning_rate', learning_rate, 'seed', seed, 'config', cfg, ...
                'scheme', 'proposed', 'resource_search', 'heuristic');
        otherwise
            out = proposed_algorithm(data.users, local, 'task', 'regression', 'test_data', test_data, ...
                'rounds', rounds, 'num_rbs', num_rbs, 'local_epochs', local_epochs, ...
                'learning_rate', learning_rate, 'seed', seed, 'config', cfg);
    end
    outputs.(schemes{i}) = out;
end

x_grid = linspace(0, 1, 100).';
fig = figure('Visible', visible_flag(cfg));
plot(data.x, data.y, 'rx', 'MarkerSize', 6); hold on;
plot(x_grid, predict_model(outputs.proposed.model_state, x_grid), 'b-', 'LineWidth', 2);
plot(x_grid, predict_model(outputs.optimal.model_state, x_grid), 'm--', 'LineWidth', 2);
plot(x_grid, predict_model(outputs.baseline_a.model_state, x_grid), 'k-', 'LineWidth', 2);
plot(x_grid, predict_model(outputs.baseline_b.model_state, x_grid), 'Color', [0, 0.7, 0], 'LineWidth', 2);
xlabel('Input of the FL algorithm');
ylabel('Output of the FL algorithm');
xlim([0, 1]); ylim([-2, 2]);
legend({'Data samples', 'Proposed algorithm', 'Optimal FL', 'Baseline a)', 'Baseline b)'}, 'Location', 'best');
result = struct('figure', fig, 'x_grid', x_grid, 'samples', [data.x, data.y], ...
    'proposed_loss', outputs.proposed.metrics.loss(end), ...
    'baseline_a_loss', outputs.baseline_a.metrics.loss(end), ...
    'baseline_b_loss', outputs.baseline_b.metrics.loss(end));
end

function result = figure_4(contexts)
context = contexts{1};
cfg = context.loss;
fcfg = cfg.figures.figure_4;
sample_counts = as_vector(fcfg.sample_counts);
rounds = first_value(fcfg.rounds);
num_rbs = first_value(fcfg.num_rbs);
local_epochs = first_value(fcfg.local_epochs);
activation = char(first_value(fcfg.activation));
learning_rate = first_value(fcfg.learning_rate);
run_seeds = cfg.meta.run_seeds;
runners = {'proposed', 'baseline_a', 'baseline_b'};
curves = struct('proposed', zeros(1, numel(sample_counts)), ...
    'baseline_a', zeros(1, numel(sample_counts)), 'baseline_b', zeros(1, numel(sample_counts)));
for s = 1:numel(run_seeds)
    for c = 1:numel(sample_counts)
        data = generate_synthetic_data('num_users', 15, 'samples_per_user', sample_counts(c), 'seed', run_seeds(s));
        train_data = struct('X', data.x, 'Y', data.y);
        rng(run_seeds(s), 'twister');
        initial = RegressionFNN('activation', activation);
        for r = 1:numel(runners)
            out = call_runner(runners{r}, data.users, initial, 'regression', train_data, rounds, num_rbs, local_epochs, learning_rate, run_seeds(s), cfg);
            eval_result = evaluate_model(out.model_state, train_data, data.users, 'regression', cfg.training);
            curves.(runners{r})(c) = curves.(runners{r})(c) + eval_result.loss;
        end
    end
end
for r = 1:numel(runners)
    curves.(runners{r}) = curves.(runners{r}) ./ numel(run_seeds);
end
fig = figure('Visible', visible_flag(cfg));
plot(sample_counts, curves.proposed, 'bo-', 'LineWidth', 2); hold on;
plot(sample_counts, curves.baseline_a, 'r--', 'LineWidth', 2);
plot(sample_counts, curves.baseline_b, 'ks--', 'LineWidth', 2);
xlabel('Number of data samples per user');
ylabel('Value of the loss function');
legend({'Proposed algorithm', 'Baseline a)', 'Baseline b)'}, 'Location', 'best');
result = struct('figure', fig, 'samples_per_user', sample_counts, 'loss', curves);
end

function result = figure_5(contexts)
context = contexts{1};
cfg = context.loss;
fcfg = cfg.figures.figure_5;
users = as_vector(fcfg.user_counts);
rb_counts = as_vector(fcfg.rb_counts);
curves = struct();
for r = 1:numel(rb_counts)
    values = zeros(1, numel(users));
    for u = 1:numel(users)
        out = proposed_algorithm([], [], 'num_users', users(u), 'num_rbs', rb_counts(r), ...
            'seed', context.seed, 'config', cfg);
        values(u) = out.solver_iterations;
    end
    curves.(sprintf('R%d', rb_counts(r))) = values;
end
fig = figure('Visible', visible_flag(cfg));
styles = {'bo-', 'ks--', 'r^-.', 'gd:'};
hold on;
for r = 1:numel(rb_counts)
    plot(users, curves.(sprintf('R%d', rb_counts(r))), styles{mod(r - 1, numel(styles)) + 1}, 'LineWidth', 2);
end
xlabel('Number of users');
ylabel('Number of iterations');
legend(arrayfun(@(x) sprintf('R=%d', x), rb_counts, 'UniformOutput', false), 'Location', 'best');
result = struct('figure', fig, 'users', users, 'edge_weight_evaluations', curves);
end

function result = figure_6(contexts)
context = contexts{1};
cfg = context.loss;
fcfg = cfg.figures.figure_6;
wireless = cfg.wireless;
users = as_vector(fcfg.user_counts);
num_rbs = first_value(fcfg.num_rbs);
samples = as_vector(fcfg.samples_per_user);
model_parameters = first_value(fcfg.model_parameters);
quant_bits = first_value(cfg.training.quantization_bits);
trials = first_value(fcfg.simulation_trials);
rounds = first_value(fcfg.rounds);
mu = first_value(fcfg.strong_convexity_mu);
L = first_value(fcfg.lipschitz_l);
zeta1 = first_value(fcfg.gradient_bound_zeta1);
zeta2 = first_value(fcfg.gradient_bound_zeta2);
threshold = first_value(fcfg.waterfall_threshold);
interference = interference_for_rbs(wireless, num_rbs);
run_seeds = cfg.meta.run_seeds;
theoretical = zeros(numel(run_seeds), numel(users));
simulated = zeros(numel(run_seeds), numel(users));
selected_counts = zeros(numel(run_seeds), numel(users));

for s = 1:numel(run_seeds)
    rng(run_seeds(s), 'twister');
    distance_pool = max(rand(1, max(users)) .* wireless.radius_m, realmin);
    for u = 1:numel(users)
        count = samples(1:users(u));
        total = sum(count);
        d = distance_pool(1:users(u)).';
        path_gain = d .^ (-wireless.path_loss_alpha);
        denom = reshape(interference, 1, []) + 1e-14;
        q = 1 - exp(-threshold .* denom ./ max(wireless.pmax_w .* path_gain, realmin));
        uplink_rate = log2(1 + wireless.pmax_w .* path_gain ./ max(denom, realmin));
        downlink_rate = (wireless.downlink_bandwidth_hz / 1e6) .* log2(1 + path_gain ./ max(wireless.downlink_interference_w, realmin));
        model_mb = model_parameters * quant_bits / 1024 / 1024;
        delay = model_mb ./ max(uplink_rate, realmin) + model_mb ./ max(downlink_rate, realmin);
        energy = wireless.energy_coefficient * wireless.cpu_cycles_per_bit * wireless.cpu_frequency_hz^2 * model_mb + wireless.pmax_w .* (model_mb ./ max(uplink_rate, realmin));
        feasible = delay < wireless.delay_s & energy < wireless.energy_j;
        weights = zeros(users(u), num_rbs);
        for i = 1:users(u)
            weights(i, :) = count(i) .* (q(i, :) - 1) .* feasible(i, :);
        end
        assignment = assignment_from_weights(weights);
        final_q = ones(1, users(u));
        selected = [];
        for i = 1:users(u)
            if assignment(i) > 0 && feasible(i, assignment(i)) && weights(i, assignment(i)) < 0
                final_q(i) = q(i, assignment(i));
                selected(end + 1) = i; %#ok<AGROW>
            end
        end
        miss_ratio = sum(count .* final_q) / total;
        contraction = 1 - mu / L + 4 * mu * zeta2 * miss_ratio / L;
        theoretical(s, u) = (2 * zeta1 * miss_ratio / L) / max(1 - contraction, 1e-9);
        selected_counts(s, u) = numel(selected);
        gap = zeros(trials, 1);
        if isempty(selected)
            simulated(s, u) = theoretical(s, u);
        else
            selected_errors = final_q(selected);
            selected_weight = count(selected);
            for t = 1:rounds
                successes = rand(trials, numel(selected)) > selected_errors;
                missed = total - successes * selected_weight(:);
                miss = missed ./ total;
                gap = (1 - mu / L + 4 * mu * zeta2 .* miss / L) .* gap + 2 * zeta1 .* miss / L;
            end
            simulated(s, u) = mean(gap);
        end
    end
end

fig = figure('Visible', visible_flag(cfg));
plot(users, mean(theoretical, 1), 'bo-', 'LineWidth', 2); hold on;
plot(users, mean(simulated, 1), 'ks--', 'LineWidth', 2);
xlabel('Number of users');
ylabel('Convergence gap due to wireless factors');
legend({'Theoretical analysis', 'Simulation result'}, 'Location', 'best');
result = struct('figure', fig, 'users', users, 'theoretical_gap', mean(theoretical, 1), ...
    'simulation_gap', mean(simulated, 1), 'selected_users', mean(selected_counts, 1));
end

function result = figure_7(contexts)
context = contexts{1};
cfg = context.loss;
fcfg = cfg.figures.figure_7;
samples = as_vector(fcfg.samples_per_user);
rounds = first_value(fcfg.rounds);
num_rbs = first_value(fcfg.num_rbs);
local_epochs = first_value(fcfg.local_epochs);
learning_rate = first_value(fcfg.learning_rate);
test_samples = first_value(fcfg.test_samples);
run_seeds = cfg.meta.run_seeds;
runners = {'proposed', 'baseline_a', 'baseline_b', 'baseline_c'};
curves = struct();
for r = 1:numel(runners)
    curves.(runners{r}) = zeros(numel(run_seeds), rounds);
end
for s = 1:numel(run_seeds)
    data = load_mnist_data('num_users', numel(samples), 'samples_per_user', samples, ...
        'test_samples', test_samples, 'seed', run_seeds(s), ...
        'train_order', cfg.training.mnist_train_order, 'partition_order', cfg.training.mnist_partition_order);
    rng(run_seeds(s), 'twister');
    initial = MNISTFNN('activation', cfg.training.mnist_activation);
    for r = 1:numel(runners)
        out = call_runner(runners{r}, data.users, initial, 'mnist', data.test, rounds, num_rbs, local_epochs, learning_rate, run_seeds(s), cfg);
        curves.(runners{r})(s, :) = out.metrics.accuracy;
    end
end
round_axis = 1:rounds;
fig = figure('Visible', visible_flag(cfg));
plot(round_axis, mean(curves.proposed, 1), 'k-', 'LineWidth', 2); hold on;
plot(round_axis, mean(curves.baseline_a, 1), 'b--', 'LineWidth', 2);
plot(round_axis, mean(curves.baseline_b, 1), 'r-', 'LineWidth', 2);
plot(round_axis, mean(curves.baseline_c, 1), 'r:', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Identification accuracy');
legend({'Proposed FL', 'Baseline a)', 'Baseline b)', 'Baseline c)'}, 'Location', 'best');
result = struct('figure', fig, 'rounds', round_axis, 'accuracy', curves);
end

function result = figure_8(contexts)
context = contexts{1};
cfg = context.loss;
fcfg = cfg.figures.figure_8;
user_counts = as_vector(fcfg.user_counts);
sample_pool = as_vector(fcfg.samples_per_user);
rounds = first_value(fcfg.rounds);
num_rbs = first_value(fcfg.num_rbs);
local_epochs = first_value(fcfg.local_epochs);
learning_rate = first_value(fcfg.learning_rate);
test_samples = first_value(fcfg.test_samples);
runners = {'proposed', 'baseline_a', 'baseline_b', 'baseline_c'};
curves = struct('proposed', zeros(1, numel(user_counts)), 'baseline_a', zeros(1, numel(user_counts)), ...
    'baseline_b', zeros(1, numel(user_counts)), 'baseline_c', zeros(1, numel(user_counts)));
for u = 1:numel(user_counts)
    samples = sample_pool(1:user_counts(u));
    data = load_mnist_data('num_users', user_counts(u), 'samples_per_user', samples, ...
        'test_samples', test_samples, 'seed', context.seed, ...
        'train_order', cfg.training.mnist_train_order, 'partition_order', cfg.training.mnist_partition_order);
    rng(context.seed, 'twister');
    initial = MNISTFNN('activation', cfg.training.mnist_activation);
    for r = 1:numel(runners)
        out = call_runner(runners{r}, data.users, initial, 'mnist', data.test, rounds, num_rbs, local_epochs, learning_rate, context.seed, cfg);
        curves.(runners{r})(u) = out.metrics.accuracy(end);
    end
end
fig = figure('Visible', visible_flag(cfg));
plot(user_counts, curves.proposed, 'k-', 'LineWidth', 2); hold on;
plot(user_counts, curves.baseline_a, 'b--', 'LineWidth', 2);
plot(user_counts, curves.baseline_b, 'r-', 'LineWidth', 2);
plot(user_counts, curves.baseline_c, 'r:', 'LineWidth', 2);
xlabel('Total number of users');
ylabel('Identification accuracy');
legend({'Proposed FL', 'Baseline a)', 'Baseline b)', 'Baseline c)'}, 'Location', 'best');
result = struct('figure', fig, 'users', user_counts, 'accuracy', curves);
end

function result = figure_9(contexts)
context = contexts{1};
cfg = context.loss;
fcfg = cfg.figures.figure_9;
rb_counts = as_vector(fcfg.rb_counts);
samples = as_vector(fcfg.samples_per_user);
rounds = first_value(fcfg.rounds);
local_epochs = first_value(fcfg.local_epochs);
learning_rate = first_value(fcfg.learning_rate);
test_samples = first_value(fcfg.test_samples);
runners = {'proposed', 'baseline_a', 'baseline_b', 'baseline_c'};
curves = struct('proposed', zeros(1, numel(rb_counts)), 'baseline_a', zeros(1, numel(rb_counts)), ...
    'baseline_b', zeros(1, numel(rb_counts)), 'baseline_c', zeros(1, numel(rb_counts)));
data = load_mnist_data('num_users', numel(samples), 'samples_per_user', samples, ...
    'test_samples', test_samples, 'seed', context.seed, ...
    'train_order', cfg.training.mnist_train_order, 'partition_order', cfg.training.mnist_partition_order);
rng(context.seed, 'twister');
initial = MNISTFNN('activation', cfg.training.mnist_activation);
for rb = 1:numel(rb_counts)
    for r = 1:numel(runners)
        out = call_runner(runners{r}, data.users, initial, 'mnist', data.test, rounds, rb_counts(rb), local_epochs, learning_rate, context.seed, cfg);
        curves.(runners{r})(rb) = out.metrics.accuracy(end);
    end
end
fig = figure('Visible', visible_flag(cfg));
plot(rb_counts, curves.proposed, 'k-', 'LineWidth', 2); hold on;
plot(rb_counts, curves.baseline_a, 'b--', 'LineWidth', 2);
plot(rb_counts, curves.baseline_b, 'r-', 'LineWidth', 2);
plot(rb_counts, curves.baseline_c, 'r:', 'LineWidth', 2);
xlabel('Number of RBs');
ylabel('Identification accuracy');
legend({'Proposed FL', 'Baseline a)', 'Baseline b)', 'Baseline c)'}, 'Location', 'best');
result = struct('figure', fig, 'rbs', rb_counts, 'accuracy', curves);
end

function result = figure_10(contexts)
context = contexts{1};
cfg = context.loss;
fcfg = cfg.figures.figure_10;
samples_per_user = first_value(fcfg.samples_per_user);
test_samples = first_value(fcfg.test_samples);
rounds = first_value(fcfg.rounds);
num_rbs = first_value(fcfg.num_rbs);
local_epochs = first_value(fcfg.local_epochs);
learning_rate = first_value(fcfg.learning_rate);
data = load_mnist_data('num_users', 15, 'samples_per_user', samples_per_user, ...
    'test_samples', test_samples, 'seed', context.seed, ...
    'train_order', cfg.training.mnist_train_order, 'partition_order', cfg.training.mnist_partition_order);
rng(context.seed, 'twister');
initial = MNISTCNN();
proposed = proposed_algorithm(data.users, initial, 'task', 'mnist', 'test_data', data.test, ...
    'rounds', rounds, 'num_rbs', num_rbs, 'local_epochs', local_epochs, ...
    'learning_rate', learning_rate, 'seed', context.seed, 'config', cfg);
rng(context.seed, 'twister');
baseline = baseline_b(data.users, initial, 'task', 'mnist', 'test_data', data.test, ...
    'rounds', rounds, 'num_rbs', num_rbs, 'local_epochs', local_epochs, ...
    'learning_rate', learning_rate, 'seed', context.seed, 'config', cfg);
[~, proposed_pred] = predict_model(proposed.model_state, data.test);
[~, baseline_pred] = predict_model(baseline.model_state, data.test);
labels = data.test.Y(:).';
display_count = min(36, numel(labels));
proposed_correct = sum(proposed_pred(1:display_count) == labels(1:display_count));
baseline_correct = sum(baseline_pred(1:display_count) == labels(1:display_count));

fig = figure('Visible', visible_flag(cfg));
for i = 1:display_count
    subplot(6, 6, i);
    image = squeeze(data.test.X4(:, :, 1, i));
    imagesc(image);
    colormap(gca, 'gray');
    axis image off;
    title(sprintf('%d/%d', proposed_pred(i) - 1, baseline_pred(i) - 1), 'FontSize', 8);
end
summary_title = sprintf('Proposed FL: %d, Baseline b): %d', proposed_correct, baseline_correct);
if exist('sgtitle', 'file') == 2
    sgtitle(summary_title);
else
    subplot(6, 6, 1);
    title(summary_title, 'FontSize', 8);
end
result = struct('figure', fig, 'proposed_accuracy', proposed.metrics.accuracy(end), ...
    'baseline_b_accuracy', baseline.metrics.accuracy(end), ...
    'proposed_correct', proposed_correct, 'baseline_b_correct', baseline_correct, ...
    'display_count', display_count);
end

function out = call_runner(name, users, initial, task, test_data, rounds, num_rbs, local_epochs, learning_rate, seed, cfg)
switch name
    case 'baseline_a'
        out = baseline_a(users, initial, 'task', task, 'test_data', test_data, 'rounds', rounds, ...
            'num_rbs', num_rbs, 'local_epochs', local_epochs, 'learning_rate', learning_rate, ...
            'seed', seed, 'config', cfg);
    case 'baseline_b'
        out = baseline_b(users, initial, 'task', task, 'test_data', test_data, 'rounds', rounds, ...
            'num_rbs', num_rbs, 'local_epochs', local_epochs, 'learning_rate', learning_rate, ...
            'seed', seed, 'config', cfg);
    case 'baseline_c'
        out = baseline_c(users, initial, 'task', task, 'test_data', test_data, 'rounds', rounds, ...
            'num_rbs', num_rbs, 'local_epochs', local_epochs, 'learning_rate', learning_rate, ...
            'seed', seed, 'config', cfg);
    otherwise
        out = proposed_algorithm(users, initial, 'task', task, 'test_data', test_data, 'rounds', rounds, ...
            'num_rbs', num_rbs, 'local_epochs', local_epochs, 'learning_rate', learning_rate, ...
            'seed', seed, 'config', cfg);
end
end

function cfg = default_config()
cfg = struct();
cfg.seed = 42;
cfg.wireless = struct( ...
    'radius_m', 500.0, ...
    'path_loss_alpha', 2.0, ...
    'bs_power_w', 1.0, ...
    'waterfall_threshold', 10^(0.023 / 10), ...
    'rayleigh_mean', 1.0, ...
    'channel_model', 'mean', ...
    'cpu_frequency_hz', 1e9, ...
    'energy_coefficient', 1e-27, ...
    'cpu_cycles_per_bit', 40.0, ...
    'noise_dbm_hz', -174.0, ...
    'downlink_bandwidth_hz', 20e6, ...
    'uplink_bandwidth_hz', 1e6, ...
    'pmax_w', 0.01, ...
    'ki_cycle', [12, 10, 8, 4, 2], ...
    'delay_s', 0.5, ...
    'energy_j', 0.003, ...
    'interference_w', 3e-8, ...
    'interference_w_by_rbs', struct(), ...
    'downlink_interference_w', 1.8e-7, ...
    'min_distance_m', 0.0, ...
    'interference_lognormal_sigma', 0.0, ...
    'rate_floor', 1e-12, ...
    'channel_gain_floor', 1e-18, ...
    'snr_denominator_floor', 1e-18, ...
    'power_floor_w', 1e-12, ...
    'power_solver_maxiter', 50, ...
    'heuristic_max_rbs', 20);
cfg.training = struct( ...
    'device', 'auto', ...
    'optimizer', 'gradient_descent', ...
    'quantization_bits', 16, ...
    'eval_batch_size', 256, ...
    'mnist_activation', 'relu', ...
    'mnist_train_order', 'shuffled', ...
    'mnist_partition_order', 'shuffled', ...
    'force_first_round_success', false, ...
    'regression_loss', 'nmse', ...
    'regression_scale_floor', 1e-12, ...
    'regression_lr', 0.08, ...
    'mnist_lr', 0.08);
cfg.figures = struct();
cfg.figures.figure_3 = struct('data_count', 50, 'test_count', 1000, 'rounds', 80, ...
    'num_rbs', 12, 'local_epochs', 1, 'activation', 'tanh', 'learning_rate', 0.01);
cfg.figures.figure_4 = struct('sample_counts', [10, 20, 30, 40, 50], 'rounds', 60, ...
    'num_rbs', 12, 'local_epochs', 1, 'activation', 'tanh', 'learning_rate', 0.01);
cfg.figures.figure_5 = struct('user_counts', [3, 6, 10, 15, 20, 25], 'rb_counts', [10, 15]);
cfg.figures.figure_6 = struct('user_counts', [3, 6, 9, 12, 15, 18], 'num_rbs', 12, ...
    'samples_per_user', [100, 200, 300, 400, 500, 400, 300, 200, 100, 200, 300, 400, 500, 600, 100, 200, 300, 400], ...
    'model_parameters', 39760, 'waterfall_threshold', 1.08, 'simulation_trials', 2000, ...
    'rounds', 130, 'strong_convexity_mu', 0.1, 'lipschitz_l', 1.0, ...
    'gradient_bound_zeta1', 1.0, 'gradient_bound_zeta2', 0.5);
cfg.figures.figure_7 = struct('samples_per_user', [100, 200, 300, 400, 500, 400, 300, 200, 100, 200, 300, 400, 500, 600, 100], ...
    'test_samples', 1000, 'rounds', 130, 'num_rbs', 12, 'local_epochs', 1, 'learning_rate', 0.08);
cfg.figures.figure_8 = struct('user_counts', [3, 6, 9, 12, 15, 18], ...
    'samples_per_user', [100, 200, 300, 400, 500, 400, 300, 200, 100, 200, 300, 400, 500, 600, 100, 200, 300, 400], ...
    'test_samples', 1000, 'rounds', 130, 'num_rbs', 12, 'local_epochs', 1, 'learning_rate', 0.08);
cfg.figures.figure_9 = struct('rb_counts', [3, 6, 9, 12], ...
    'samples_per_user', [100, 200, 300, 400, 500, 400, 300, 200, 100, 200, 300, 400, 500, 600, 100], ...
    'test_samples', 1000, 'rounds', 130, 'local_epochs', 1, 'learning_rate', 0.08, 'runner_seed_mode', 'same');
cfg.figures.figure_10 = struct('samples_per_user', 2000, 'test_samples', 36, ...
    'rounds', 130, 'num_rbs', 12, 'local_epochs', 1, 'learning_rate', 0.08);
end

function [partitions, model, opts] = parse_algorithm_inputs(varargin)
idx = 1;
partitions = [];
model = [];
if idx <= numel(varargin) && (iscell(varargin{idx}) || isempty(varargin{idx}))
    partitions = varargin{idx};
    idx = idx + 1;
end
if idx <= numel(varargin) && ~(ischar(varargin{idx}) || isstring(varargin{idx}))
    model = varargin{idx};
    idx = idx + 1;
end
opts = struct('task', 'regression', 'test_data', [], 'num_users', 15, 'num_rbs', 12, ...
    'config', [], 'rounds', 0, 'local_epochs', 1, 'batch_size', 32, ...
    'seed', [], 'learning_rate', [], 'model_bits', [], 'scheme', 'proposed', ...
    'resource_search', 'hungarian');
while idx <= numel(varargin)
    key = lower(char(varargin{idx}));
    value = varargin{idx + 1};
    key = strrep(key, '-', '_');
    opts.(key) = value;
    idx = idx + 2;
end
if isempty(opts.config)
    opts.config = default_config();
end
if isfield(opts.config, 'seed') && isempty(opts.seed)
    opts.seed = first_value(opts.config.seed);
end
opts.seed = double(opts.seed);
end

function link = compute_wireless_links(wireless, num_users, num_rbs, model_bits, seed)
rng(seed, 'twister');
radius = wireless.radius_m;
distances = max(wireless.min_distance_m, radius .* sqrt(rand(num_users, 1)));
switch lower(wireless.channel_model)
    case 'mean'
        channel_gain = wireless.rayleigh_mean .* ones(num_users, num_rbs) .* distances .^ (-wireless.path_loss_alpha);
        downlink_gain = wireless.rayleigh_mean .* distances .^ (-wireless.path_loss_alpha);
    case 'rayleigh'
        channel_gain = (-wireless.rayleigh_mean .* log(max(rand(num_users, num_rbs), realmin))) .* distances .^ (-wireless.path_loss_alpha);
        downlink_gain = (-wireless.rayleigh_mean .* log(max(rand(num_users, 1), realmin))) .* distances .^ (-wireless.path_loss_alpha);
    otherwise
        error('wireless.channel_model must be mean or rayleigh');
end
n0 = 10^((wireless.noise_dbm_hz - 30) / 10);
interference = interference_for_rbs(wireless, num_rbs);
downlink_interference = wireless.downlink_interference_w;
downlink_rates = wireless.downlink_bandwidth_hz .* log2(1 + wireless.bs_power_w .* downlink_gain ./ ...
    (downlink_interference + wireless.downlink_bandwidth_hz .* n0));
downlink_delays = model_bits ./ max(downlink_rates, wireless.rate_floor);

powers = zeros(num_users, num_rbs);
packet_errors = ones(num_users, num_rbs);
uplink_rates = zeros(num_users, num_rbs);
total_delays = inf(num_users, num_rbs);
energies = inf(num_users, num_rbs);
feasible = false(num_users, num_rbs);
model_megabits = model_bits / 1024 / 1024;
train_energy = wireless.energy_coefficient * wireless.cpu_cycles_per_bit * wireless.cpu_frequency_hz^2 * model_megabits;
for u = 1:num_users
    for r = 1:num_rbs
        gain = max(channel_gain(u, r), wireless.channel_gain_floor);
        noise_i = interference(r) + wireless.uplink_bandwidth_hz * n0;
        energy_at = @(p) train_energy + p * model_bits ./ max(wireless.uplink_bandwidth_hz .* log2(1 + p .* gain ./ noise_i), wireless.rate_floor);
        if train_energy >= wireless.energy_j
            power = min(wireless.pmax_w, wireless.power_floor_w);
        elseif energy_at(wireless.pmax_w) <= wireless.energy_j
            power = wireless.pmax_w;
        elseif energy_at(wireless.power_floor_w) > wireless.energy_j
            power = wireless.power_floor_w;
        else
            lo = wireless.power_floor_w;
            hi = wireless.pmax_w;
            for it = 1:wireless.power_solver_maxiter %#ok<NASGU>
                mid = (lo + hi) / 2;
                if energy_at(mid) > wireless.energy_j
                    hi = mid;
                else
                    lo = mid;
                end
            end
            power = (lo + hi) / 2;
        end
        rate = wireless.uplink_bandwidth_hz * log2(1 + power * gain / noise_i);
        q = 1 - exp(-wireless.waterfall_threshold * noise_i / max(power * gain, wireless.snr_denominator_floor));
        q = min(1, max(0, q));
        delay = model_bits / max(rate, wireless.rate_floor);
        energy = energy_at(power);
        powers(u, r) = power;
        packet_errors(u, r) = q;
        uplink_rates(u, r) = rate;
        total_delays(u, r) = delay + downlink_delays(u);
        energies(u, r) = energy;
        feasible(u, r) = total_delays(u, r) <= wireless.delay_s && energy <= wireless.energy_j;
    end
end
link = struct('distances', distances, 'channel_gain', channel_gain, 'downlink_gain', downlink_gain, ...
    'interference', interference, 'downlink_rates', downlink_rates, 'powers', powers, ...
    'packet_errors', packet_errors, 'uplink_rates', uplink_rates, 'total_delays', total_delays, ...
    'energies', energies, 'feasible', feasible);
end

function [allocation, selected_users, assigned_rbs, iterations] = solve_assignment(scheme, link, counts)
[num_users, num_rbs] = size(link.packet_errors);
iterations = num_users * num_rbs;
allocation = zeros(num_users, num_rbs);
selected_users = [];
assigned_rbs = [];
switch lower(scheme)
    case {'proposed', 'optimal_fl'}
        weights = zeros(num_users, num_rbs);
        for u = 1:num_users
            weights(u, :) = counts(u) .* (link.packet_errors(u, :) - 1) .* link.feasible(u, :);
        end
        assignment = assignment_from_weights(weights);
        for u = 1:num_users
            r = assignment(u);
            if r > 0 && link.feasible(u, r) && weights(u, r) < 0
                allocation(u, r) = 1;
                selected_users(end + 1) = u; %#ok<AGROW>
                assigned_rbs(end + 1) = r; %#ok<AGROW>
            end
        end
    case 'baseline_a'
        weights = zeros(num_users, num_rbs);
        for u = 1:num_users
            weights(u, :) = counts(u) .* (link.packet_errors(u, :) - 1) .* link.feasible(u, :);
        end
        assignment = assignment_from_weights(weights);
        assignment_users = find(assignment > 0);
        random_rbs = randperm(num_rbs, min(numel(assignment_users), num_rbs));
        for k = 1:numel(random_rbs)
            u = assignment_users(k);
            r = random_rbs(k);
            allocation(u, r) = 1;
            selected_users(end + 1) = u; %#ok<AGROW>
            assigned_rbs(end + 1) = r; %#ok<AGROW>
        end
    case 'baseline_b'
        if num_rbs < num_users
            selected_users = randperm(num_users, num_rbs);
        else
            selected_users = 1:num_users;
        end
        assigned_rbs = randperm(num_rbs, numel(selected_users));
        for k = 1:numel(selected_users)
            allocation(selected_users(k), assigned_rbs(k)) = 1;
        end
    case 'baseline_c'
        weights = (link.packet_errors - 1) .* link.feasible;
        assignment = assignment_from_weights(weights);
        for u = 1:num_users
            r = assignment(u);
            if r > 0 && link.feasible(u, r) && weights(u, r) < 0
                allocation(u, r) = 1;
                selected_users(end + 1) = u; %#ok<AGROW>
                assigned_rbs(end + 1) = r; %#ok<AGROW>
            end
        end
    otherwise
        error('Unsupported scheme: %s', scheme);
end
end

function assignment = assignment_from_weights(weights)
if isempty(weights)
    assignment = [];
    return;
end
cost = weights - min(weights(:));
if exist('munkres', 'file') == 2
    assignment = munkres(cost);
else
    [num_users, num_rbs] = size(weights);
    assignment = zeros(1, num_users);
    used = false(1, num_rbs);
    for k = 1:min(num_users, num_rbs)
        best = inf;
        bestu = 0;
        bestr = 0;
        for u = 1:num_users
            if assignment(u) > 0
                continue;
            end
            for r = 1:num_rbs
                if ~used(r) && cost(u, r) < best
                    best = cost(u, r);
                    bestu = u;
                    bestr = r;
                end
            end
        end
        if bestu > 0
            assignment(bestu) = bestr;
            used(bestr) = true;
        end
    end
end
end

function model = train_full_batch(model, data, task, lr, train_cfg)
switch model.type
    case {'regression_fnn', 'mnist_fnn', 'mnist_cnn_fallback'}
        [prediction, cache] = forward_manual(model, data.X);
        n = size(data.X, 1);
        if strcmp(task, 'regression')
            target = data.Y(:);
            error_value = prediction(:) - target;
            if strcmpi(train_cfg.regression_loss, 'nmse')
                scale = max(max(target) - min(target), train_cfg.regression_scale_floor);
                dY = 8 .* error_value ./ (n * scale^2);
            else
                dY = 2 .* error_value ./ n;
            end
            dY = reshape(dY, [], 1);
        else
            logits = prediction;
            logits = logits - max(logits, [], 2);
            expv = exp(logits);
            probs = expv ./ sum(expv, 2);
            labels = data.Y(:);
            dY = probs;
            dY(sub2ind(size(dY), (1:n).', labels)) = dY(sub2ind(size(dY), (1:n).', labels)) - 1;
            dY = dY ./ n;
        end
        grads = backward_manual(model, cache, dY);
        names = fieldnames(model.params);
        for i = 1:numel(names)
            model.params.(names{i}) = model.params.(names{i}) - lr .* grads.(names{i});
        end
    case 'mnist_dlnetwork'
        dlX = dlarray(single(data.X4), 'SSCB');
        labels = data.Y(:).';
        T = zeros(10, numel(labels), 'single');
        T(sub2ind(size(T), labels, 1:numel(labels))) = 1;
        dlT = dlarray(T, 'CB');
        [gradients, ~] = dlfeval(@cnn_gradients, model.net, dlX, dlT);
        model.net = dlupdate(@(w, g) w - lr .* g, model.net, gradients);
    otherwise
        error('Unsupported model type: %s', model.type);
end
end

function [gradients, loss] = cnn_gradients(net, dlX, dlT)
scores = forward(net, dlX);
probs = softmax(scores);
loss = crossentropy(probs, dlT);
gradients = dlgradient(loss, net.Learnables);
end

function [Y, cache] = forward_manual(model, X)
Z1 = X * model.params.W1.' + model.params.b1.';
A1 = activate(Z1, model.activation);
Y = A1 * model.params.W2.' + model.params.b2.';
cache = struct('X', X, 'Z1', Z1, 'A1', A1);
end

function grads = backward_manual(model, cache, dY)
grads = struct();
grads.W2 = dY.' * cache.A1;
grads.b2 = sum(dY, 1).';
dA1 = dY * model.params.W2;
dZ1 = dA1 .* activate_grad(cache.Z1, model.activation);
grads.W1 = dZ1.' * cache.X;
grads.b1 = sum(dZ1, 1).';
end

function A = activate(Z, activation)
switch lower(activation)
    case 'tanh'
        A = tanh(Z);
    case 'sigmoid'
        A = 1 ./ (1 + exp(-Z));
    case 'identity'
        A = Z;
    otherwise
        A = max(0, Z);
end
end

function G = activate_grad(Z, activation)
switch lower(activation)
    case 'tanh'
        G = 1 - tanh(Z).^2;
    case 'sigmoid'
        S = 1 ./ (1 + exp(-Z));
        G = S .* (1 - S);
    case 'identity'
        G = ones(size(Z));
    otherwise
        G = double(Z > 0);
end
end

function model = average_models(models, weights)
weights = weights ./ sum(weights);
model = models{1};
switch model.type
    case {'regression_fnn', 'mnist_fnn', 'mnist_cnn_fallback'}
        names = fieldnames(model.params);
        for n = 1:numel(names)
            acc = zeros(size(model.params.(names{n})));
            for i = 1:numel(models)
                acc = acc + weights(i) .* models{i}.params.(names{n});
            end
            model.params.(names{n}) = acc;
        end
    case 'mnist_dlnetwork'
        learnables = model.net.Learnables;
        for row = 1:size(learnables, 1)
            value = weights(1) .* models{1}.net.Learnables.Value{row};
            for i = 2:numel(models)
                value = value + weights(i) .* models{i}.net.Learnables.Value{row};
            end
            learnables.Value{row} = value;
        end
        model.net.Learnables = learnables;
end
end

function eval_result = evaluate_model(model, test_data, partitions, task, train_cfg)
if isempty(test_data)
    X = [];
    Y = [];
    X4 = [];
    for i = 1:numel(partitions)
        X = [X; partitions{i}.X]; %#ok<AGROW>
        Y = [Y; partitions{i}.Y]; %#ok<AGROW>
        if isfield(partitions{i}, 'X4')
            X4 = cat(4, X4, partitions{i}.X4); %#ok<AGROW>
        end
    end
    test_data = struct('X', X, 'Y', Y);
    if ~isempty(X4)
        test_data.X4 = X4;
    end
end
[pred, labels] = predict_model(model, test_data);
if strcmp(task, 'regression')
    target = test_data.Y(:);
    mse = mean((pred(:) - target).^2);
    if strcmpi(train_cfg.regression_loss, 'nmse')
        scale = max(max(target) - min(target), train_cfg.regression_scale_floor);
        loss = 4 * mse / (scale^2);
    else
        loss = mse;
    end
    accuracy = NaN;
else
    target = test_data.Y(:).';
    probs = softmax_rows(pred);
    idx = sub2ind(size(probs), (1:numel(target)).', target(:));
    loss = -mean(log(max(probs(idx), realmin)));
    accuracy = mean(labels == target);
end
eval_result = struct('loss', loss, 'accuracy', accuracy);
end

function [prediction, labels] = predict_model(model, data)
labels = [];
if isnumeric(data)
    data = struct('X', data);
end
switch model.type
    case {'regression_fnn', 'mnist_fnn', 'mnist_cnn_fallback'}
        [prediction, ~] = forward_manual(model, data.X);
        if ~strcmp(model.type, 'regression_fnn')
            [~, labels] = max(prediction, [], 2);
            labels = labels(:).';
        end
    case 'mnist_dlnetwork'
        dlX = dlarray(single(data.X4), 'SSCB');
        scores = extractdata(forward(model.net, dlX));
        prediction = double(scores).';
        [~, labels] = max(prediction, [], 2);
        labels = labels(:).';
    otherwise
        error('Unsupported model type: %s', model.type);
end
end

function n = count_model_parameters(model)
switch model.type
    case {'regression_fnn', 'mnist_fnn', 'mnist_cnn_fallback'}
        n = 0;
        names = fieldnames(model.params);
        for i = 1:numel(names)
            n = n + numel(model.params.(names{i}));
        end
    case 'mnist_dlnetwork'
        n = 0;
        values = model.net.Learnables.Value;
        for i = 1:numel(values)
            n = n + numel(values{i});
        end
end
end

function probs = softmax_rows(logits)
logits = logits - max(logits, [], 2);
expv = exp(logits);
probs = expv ./ sum(expv, 2);
end

function [x_train, y_train, x_test, y_test] = load_npz_mnist(path)
try
    np = py.importlib.import_module('numpy');
    loaded = np.load(path);
    x_train = np_images_to_mat(loaded{'x_train'});
    y_train = np_vector_to_mat(loaded{'y_train'}) + 1;
    x_test = np_images_to_mat(loaded{'x_test'});
    y_test = np_vector_to_mat(loaded{'y_test'}) + 1;
catch err
    error('Unable to load %s through Python numpy: %s', path, err.message);
end
end

function [x_train, y_train, x_test, y_test] = load_mat_mnist(path)
loaded = load(path);
x_train = double(loaded.x_train);
y_train = double(loaded.y_train(:)) + 1;
x_test = double(loaded.x_test);
y_test = double(loaded.y_test(:)) + 1;
if max(x_train(:)) > 1
    x_train = x_train ./ 255.0;
end
if max(x_test(:)) > 1
    x_test = x_test ./ 255.0;
end
end

function images = np_images_to_mat(arr)
np = py.importlib.import_module('numpy');
arraymod = py.importlib.import_module('array');
arr = np.asarray(arr).astype('float64');
shape = cellfun(@double, cell(arr.shape));
flat = double(arraymod.array('d', np.ravel(arr)));
if numel(shape) == 3
    n = shape(1);
    tmp = reshape(flat, [shape(3), shape(2), n]);
    images = permute(tmp, [3, 2, 1]);
else
    error('Expected MNIST image array with shape N x 28 x 28');
end
if max(images(:)) > 1
    images = images ./ 255.0;
end
end

function values = np_vector_to_mat(arr)
np = py.importlib.import_module('numpy');
arraymod = py.importlib.import_module('array');
arr = np.asarray(arr).astype('float64');
values = double(arraymod.array('d', np.ravel(arr)));
values = values(:);
end

function X = flatten_mnist(images)
n = size(images, 1);
X = reshape(permute(images, [2, 3, 1]), 28 * 28, n).';
end

function X4 = to_mnist4d(images)
n = size(images, 1);
X4 = reshape(permute(images, [2, 3, 1]), 28, 28, 1, n);
end

function values = as_vector(value)
if iscell(value)
    if numel(value) == 1
        values = as_vector(value{1});
    else
        values = cellfun(@double, value);
    end
elseif isnumeric(value) || islogical(value)
    values = double(value(:)).';
else
    values = value;
end
end

function value = first_value(value)
if iscell(value)
    value = value{1};
elseif isnumeric(value) || islogical(value)
    if numel(value) > 1
        value = value(1);
    end
end
end

function value = visible_flag(cfg)
if isfield(cfg, 'meta') && isfield(cfg.meta, 'plot') && cfg.meta.plot
    value = 'on';
else
    value = 'off';
end
end

function interference = interference_for_rbs(wireless, num_rbs)
key = make_valid_name(num2str(num_rbs));
if isfield(wireless.interference_w_by_rbs, key)
    interference = as_vector(wireless.interference_w_by_rbs.(key));
else
    interference = repmat(wireless.interference_w, 1, num_rbs);
end
if numel(interference) ~= num_rbs
    error('wireless.interference_w_by_rbs[%d] must contain %d values', num_rbs, num_rbs);
end
interference = max(double(interference), 0);
end

function name = make_valid_name(text)
name = regexprep(char(text), '[^A-Za-z0-9_]', '_');
if isempty(name) || ~isletter(name(1))
    name = ['x' name];
end
name = regexprep(name, '_+', '_');
end

function tf = starts_with(text, prefix)
text = char(text);
prefix = char(prefix);
tf = numel(text) >= numel(prefix) && strcmp(text(1:numel(prefix)), prefix);
end

function tf = ends_with(text, suffix)
text = char(text);
suffix = char(suffix);
tf = numel(text) >= numel(suffix) && strcmp(text(end - numel(suffix) + 1:end), suffix);
end

function base = merge_structs(base, update)
names = fieldnames(update);
for i = 1:numel(names)
    name = names{i};
    if isfield(base, name) && isstruct(base.(name)) && isstruct(update.(name))
        base.(name) = merge_structs(base.(name), update.(name));
    else
        base.(name) = update.(name);
    end
end
end

function value = collapse_singletons(value)
if isstruct(value)
    names = fieldnames(value);
    for i = 1:numel(names)
        value.(names{i}) = collapse_singletons(value.(names{i}));
    end
elseif iscell(value)
    for i = 1:numel(value)
        value{i} = collapse_singletons(value{i});
    end
    if numel(value) == 1
        value = value{1};
    end
end
end

function parsed = parse_yaml(path)
text = fileread(path);
lines = regexp(text, '\r\n|\n|\r', 'split');
parsed = struct();
path_stack = {};
indent_stack = [];
for i = 1:numel(lines)
    raw = lines{i};
    comment_pos = strfind(raw, '#');
    if ~isempty(comment_pos)
        raw = raw(1:comment_pos(1) - 1);
    end
    if isempty(strtrim(raw))
        continue;
    end
    indent = numel(raw) - numel(regexprep(raw, '^\s*', ''));
    text = strtrim(raw);
    while ~isempty(indent_stack) && indent <= indent_stack(end)
        indent_stack(end) = [];
        path_stack(end) = [];
    end
    if starts_with(text, '- ')
        item = strtrim(text(3:end));
        value = parse_yaml_scalar(item);
        parsed = append_path_value(parsed, path_stack, value);
        continue;
    end
    parts = regexp(text, '^([^:]+):(.*)$', 'tokens', 'once');
    if isempty(parts)
        continue;
    end
    key = make_valid_name(strtrim(parts{1}));
    rest = strtrim(parts{2});
    current_path = [path_stack, {key}];
    if isempty(rest)
        parsed = set_path_value(parsed, current_path, struct());
        path_stack = current_path;
        indent_stack(end + 1) = indent; %#ok<AGROW>
    else
        parsed = set_path_value(parsed, current_path, parse_yaml_scalar(rest));
    end
end
end

function value = parse_yaml_scalar(text)
text = strtrim(text);
if starts_with(text, '[') && ends_with(text, ']')
    inner = strtrim(text(2:end - 1));
    if isempty(inner)
        value = [];
        return;
    end
    parts = strtrim(strsplit(inner, ','));
    values = cell(1, numel(parts));
    numeric = true;
    logicals = true;
    for i = 1:numel(parts)
        values{i} = parse_yaml_scalar(parts{i});
        numeric = numeric && isnumeric(values{i}) && isscalar(values{i});
        logicals = logicals && islogical(values{i}) && isscalar(values{i});
    end
    if numeric
        value = cellfun(@double, values);
    elseif logicals
        value = cellfun(@logical, values);
    else
        value = values;
    end
    return;
end
lowered = lower(text);
if strcmp(lowered, 'true')
    value = true;
elseif strcmp(lowered, 'false')
    value = false;
else
    number = str2double(text);
    if ~isnan(number)
        value = number;
    else
        value = char(strtrim(regexprep(text, '^[''"]|[''"]$', '')));
    end
end
end

function s = set_path_value(s, path, value)
if numel(path) == 1
    s.(path{1}) = value;
else
    head = path{1};
    if ~isfield(s, head) || ~isstruct(s.(head))
        s.(head) = struct();
    end
    s.(head) = set_path_value(s.(head), path(2:end), value);
end
end

function s = append_path_value(s, path, value)
if isempty(path)
    error('YAML list item without parent key');
end
current = get_path_value(s, path);
if isstruct(current) && isempty(fieldnames(current))
    current = {};
end
if ~iscell(current)
    current = {current};
end
current{end + 1} = value;
s = set_path_value(s, path, current);
end

function value = get_path_value(s, path)
value = s;
for i = 1:numel(path)
    value = value.(path{i});
end
end
