    %I think I have to transpose the reshaped things but I'm not super
    %positive. It works for the mean image though so it should be good.

    fid = fopen('C:\Users\Steve\Documents\Data\EigenHot\HotorNot_topbottom\vocab.bin', 'r');
    num_components = fread(fid,1,'int');
    num_eigen = fread(fid,1,'int');
    mean = fread(fid,num_eigen,'double');
    meanImg = reshape(mean,sqrt(num_eigen),sqrt(num_eigen));
    meanImg = transpose(uint8(meanImg));
    eigenvalues = fread(fid,num_components,'double');
    eigenvectors = fread(fid,num_eigen*num_components,'double');
    eigenvectors = transpose(reshape(eigenvectors,num_eigen,num_components));
    projections = fread(fid,num_components*num_components,'double');
    projections = transpose(reshape(projections,num_components,num_components));
    labels = fread(fid,num_components,'int');
    fclose(fid);