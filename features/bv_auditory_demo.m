function [melSpecdb,gammaSpecdb]=bv_auditory_demo(file_name)


[audioIn,fs] = audioread(file_name);

% [Gcoeffs,~,~,Gloc] = gtcc(audioIn,fs);
% [Mcoeffs,delta,deltaDelta,Mloc] = mfcc(audioIn,fs);
% 
% %figure;subplot(3,1,1);imagesc(Gcoeffs);subplot(3,1,2);imagesc(Mcoeffs)
% 
% %filename = '03a02Wb.wav';
% %[f_audio,sideinfo] = wav_to_audio('', 'data_WAV/', filename);
% shiftFB = estimateTuning(audioIn);
% 
% paramPitch.winLenSTMSP = 882;
% %paramPitch.shiftFB = shiftFB;
% paramPitch.visualize = 1;
% [f_pitch,sideinfo] = ...
%     audio_to_pitch_via_FB(audioIn,paramPitch);


numBands = 40;

spec = melSpectrogram(audioIn,fs, ...
    'Window',hamming(400,'periodic'), ...
    'OverlapLength',240, ...
    'FFTLength',1024, ...
    'NumBands',numBands);

%melSpec = 20*log10(spec+eps);
melSpecdb=20*log10(spec+eps);

[D,F] = gammatonegram(audioIn,fs,0.025,0.010,numBands);
gammaSpecdb=20*log10(D+eps);


% addpath('/Users/andrew/Downloads/npy-matlab-master/npy-matlab')  
% savepath
% load('/Users/andrew/Desktop/BV_Nov_Auditory/db3v/mel/data_wav_8s_2/3/Agelaius phoeniceus/109040_1.mat')
% load('/Users/andrew/Desktop/BV_Nov_Auditory/db3v/gamma/data_wav_8s_2/3/Agelaius phoeniceus/109040_1.mat')
% cqt = readNPY('/Users/andrew/Desktop/BV_Nov_Auditory/db3v/cqt/data_wav_8s_2/3/Agelaius phoeniceus/109040_1.npy');
% 
% figure;
% subplot(311)
% %F = fs*(0:256)/512;
% imagesc(melSpecdb)
% axis xy;
% title('MelSpectrogram');
% xlabel('Time [s]');
% ylabel('Frequency [Hz]');
% subplot(312)
% imagesc(gammaSpecdb); axis xy
% title('GammatoneSpectrogram');
% xlabel('Time [s]');
% ylabel('Frequency [Hz]');
% subplot(313)
% imagesc(cqt)
% axis xy;
% title('CQT Transform');
% xlabel('Time [s]');
% ylabel('Notes');






end
