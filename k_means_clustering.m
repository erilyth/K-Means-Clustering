load('data.mat');
imgs = data.images;
features = data.gist;

correct_classification = data.images(:,3);

%Number of clusters
k = 5;

%Max_Iterations
max_it = 10;

%Replications
replicate = 50;

least_error = 1000000;
best_result = 0;

Js = zeros(replicate,1);
Means = zeros(k,size(features,2),replicate);
As = zeros(size(features,1), k, replicate);

for tt=1:replicate
    %Initialize means
    means_idx = randperm(size(features,1),k);
    means = zeros(k,size(features,2));
    A = zeros(size(features,1),k);
    for i=1:k
        means(i,:) = features(means_idx(i),:);
    end

    for t=1:max_it
        dist_cur = 0.0;
        for i=1:size(features,1)
            best_dist = 10000000000.0;
            best_idx = 0;
            for j=1:k
                A(i,j) = 0;
            end
            for j=1:k
                dist_cur = 0.0;
                for p=1:size(features,2)
                    dist_cur = dist_cur + (features(i,p) - means(j,p)).^2;
                end
                if dist_cur <= best_dist
                    best_dist = dist_cur;
                    best_idx = j;
                end
            end
            A(i,best_idx) = 1;
        end

        %Recalculate means
        for j=1:k
            img_cnt = 0;
            for img=1:size(features,1)
                if A(img,j) ~= 0
                    img_cnt = img_cnt + 1;
                end
            end
            for p=1:size(features,2)
                means(j,p)=0;
            end
            for img=1:size(features,1)
                for p=1:size(features,2)
                    means(j,p) = means(j,p) + A(img,j) * features(img,p);
                end
            end
            %Divide by number of images
            means(j,:) = means(j,:) ./ double(img_cnt);
        end
        JErr = 0.0;
        for j=1:k
            for img=1:size(features,1)
                for p=1:size(features,2)
                    JErr = JErr + A(img,j) * (features(img,p) - means(j,p)).^2;
                end
            end
        end
        disp(JErr);
    end
    disp('Iteration over');
    Js(tt) = JErr;
    Means(:,:,tt) = means;
    As(:,:,tt) = A;
    if least_error >= JErr
        least_error = JErr;
        best_result = tt;
    end
end

error = 0;
best_clustering = zeros(60,1);
best_A = As(:,:,best_result);
for i=1:size(best_A,1)
    for j=1:size(best_A,2)
        if best_A(i,j) == 1
            best_clustering(i) = j;
            if cell2mat(correct_classification(i)) ~= best_clustering(i)
                error = error + 1;
            end
        end
    end
end

disp('Best results :');
disp('Least error (J)');
disp(Js(best_result));
disp('Assignment of images to clusters');
disp(best_clustering');
disp('Amount of matches');
disp(1.0 - error./double(60));