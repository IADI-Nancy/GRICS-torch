% Check whether J is the directional derivative of E:
%   J(m) * dm  ~=  (E(m + h*dm) - E(m - h*dm)) / (2h)
%
% Requires the original MATLAB demo functions to be on path:
% create_motion_corrupted_dataset, motion_corrupted_mri_scan,
% motion_perturbation_simulator, create_sparse_motion_operator.

clear; clc;

rng(1);

N = 128;
NshotsPerNex = 8;
Nex = 3;
motion_type_flag = 2;      % 0: translations, 2: time-constrained nonrigid
kspace_sampling_flag = 1;  % interleaved

[ExactImage, Data] = create_motion_corrupted_dataset(N, NshotsPerNex, Nex, motion_type_flag, kspace_sampling_flag);
Data.display = 0;

% Use exact image as current reconstructed image (isolates J vs E consistency).
Data.ReconstructedImage = ExactImage;

% Build baseline motion model representation compatible with your simulator code.
if Data.motion_type_flag == 0
    MotionModel = [Data.XTranslationVector; Data.YTranslationVector];
elseif Data.motion_type_flag == 2
    [X,Y] = meshgrid(1:N,1:N);
    mu_x = N/2; mu_y = N/2; sigma = N/4;
    g = exp( -(X-mu_x).^2 / (2*sigma^2) ) .* exp( -(Y-mu_y).^2 / (2*sigma^2) );
    MotionModel = cat(3, g, 2*g); % Ux = alpha_x*S, Uy = alpha_y*S
else
    error('This test is intended for motion_type_flag 0 or 2.');
end

% Random perturbation direction dm
dm = randn(size(MotionModel));
dm = dm / norm(dm(:));

% Build motion operators from current model, then evaluate J*dm
Data0 = Data;
Data0 = build_motion_ops_from_model(Data0, MotionModel);
if Data0.motion_type_flag == 2
    Data0.S = Data0.XTranslationVector;
end
Jdm = motion_perturbation_simulator(dm(:), Data0);

hs = [1e-1 5e-2 1e-2 5e-3 1e-3 5e-4 1e-4];
rel_err = zeros(size(hs));

for k = 1:numel(hs)
    h = hs(k);

    Datap = build_motion_ops_from_model(Data, MotionModel + h*dm);
    Datam = build_motion_ops_from_model(Data, MotionModel - h*dm);

    Ep = motion_corrupted_mri_scan(ExactImage(:), Datap);
    Em = motion_corrupted_mri_scan(ExactImage(:), Datam);
    fd = (Ep - Em) / (2*h);

    rel_err(k) = norm(Jdm - fd) / (norm(fd) + 1e-15);
end

fprintf('\nJ derivative check: rel_err(h)\n');
for k = 1:numel(hs)
    fprintf('h=%-9.1e  rel_err=%-12.6e\n', hs(k), rel_err(k));
end

fprintf('\nSmallest-h relative error: %.6e\n', rel_err(end));
if rel_err(end) < 5e-2
    fprintf('PASS: J is consistent with directional derivative of E at tested point.\n');
else
    fprintf('FAIL: J is not sufficiently consistent with derivative of E at tested point.\n');
end

figure; loglog(hs, rel_err, 'o-'); grid on;
xlabel('h'); ylabel('relative error');
title('Directional derivative check: J vs finite difference of E');

% -------------------------------------------------------------------------
function DataOut = build_motion_ops_from_model(DataIn, MotionModel)
    DataOut = DataIn;
    N = DataIn.N;
    Nshots = DataIn.Nshots;
    for shot = 1:Nshots
        if DataIn.motion_type_flag == 0
            tx = MotionModel(1,shot);
            ty = MotionModel(2,shot);
            Ux = tx * ones(N,N);
            Uy = ty * ones(N,N);
        elseif DataIn.motion_type_flag == 2
            S = DataIn.XTranslationVector(shot);
            Ux = MotionModel(:,:,1) * S;
            Uy = MotionModel(:,:,2) * S;
        else
            error('Unsupported motion_type_flag in build_motion_ops_from_model');
        end

        %#ok<NASGU>
        DataOut.MotionOperator{shot} = create_sparse_motion_operator(Ux, Uy);
    end
end
