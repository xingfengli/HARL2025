function visualization_demo

audio='Turdus_migratorius.wav';
audio_path=['/Volumes/T9/Mac2024/CityU_Research/BV_Nov_Auditory/Ch2Matlab2','/',audio];
[melSpecdb,gammaSpecdb]=bv_auditory_demo(audio_path);

figure;
% 设置整个图中字体大小为18
set(gcf,'DefaultAxesFontSize',16);
% 设置colormap为jet
colormap('jet');

% 绘制Mel频谱图相关设置
subplot(3,2,1);
imagesc(melSpecdb);
axis xy; % 翻转纵坐标
% 设置纵坐标标题
%ylabel('Mel Bands');
% 设置横坐标标题
% xlabel('Time Frames');
% 设置colorbar标题
colorbar;
title('Melspectrogram','FontWeight','normal');

% 绘制Gamma频谱图相关设置
subplot(3,2,2);
imagesc(gammaSpecdb);
axis xy; % 翻转纵坐标
% 设置纵坐标标题
%ylabel('Gamma Bands');
% 设置横坐标标题
% xlabel('Time Frames');
% 设置colorbar标题
colorbar;
title('Gammatone spectrogram','FontWeight','normal');

% 新增的子图操作，这里delta相关操作暂简单定义变量示例，你需按实际情况替换
delta_melSpecdb = melSpecdb(2:end,:) - melSpecdb(1:end-1,:); % 示例的差分操作，可按需改
subplot(3,2,3);
imagesc(delta_melSpecdb);
axis xy;
%ylabel('Delta Mel Bands');
% xlabel('Time Frames');
colorbar;
title('Δ Melspectrogram','FontWeight','normal');

delta_gammaSpecdb = gammaSpecdb(2:end,:) - gammaSpecdb(1:end-1,:); % 示例的差分操作，可按需改
subplot(3,2,4);
imagesc(delta_gammaSpecdb);
axis xy;
%ylabel('Delta Gamma Bands');
% xlabel('Time Frames');
colorbar;
title('Δ Gammatone spectrogram','FontWeight','normal');

delta_delta_melSpecdb = delta_melSpecdb(2:end,:) - delta_melSpecdb(1:end-1,:); % 示例的差分操作，可按需改
subplot(3,2,5);
imagesc(delta_delta_melSpecdb);
axis xy;
%ylabel('Delta Delta Mel Bands');
xlabel('Time Frames');
colorbar;
title('Δ Δ Melspectrogram','FontWeight','normal');

delta_delta_gammaSpecdb = delta_gammaSpecdb(2:end,:) - delta_gammaSpecdb(1:end-1,:); % 示例的差分操作，可按需改
subplot(3,2,6);
imagesc(delta_delta_gammaSpecdb);
axis xy;
%ylabel('Delta Delta Gamma Bands');
xlabel('Time Frames');
colorbar;
title('Δ Δ Gammatone spectrogram','FontWeight','normal');

end
