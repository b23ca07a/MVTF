import numpy as np
import torch, torchaudio
import scipy
import soundfile as sf
from time import time
from utils.FRAM_RIR import FRAM_RIR

class RandomRirGenerator(object):
    def __init__(self, sound_velocity=343, fs=16000, 
                 roomdim_range_x=[5, 10], roomdim_range_y=[5, 10], roomdim_range_z=[2.5, 4.5], 
                 micpos='corner', 
                 roomcenter_mic_dist_max_x=0.5, roomcenter_mic_dist_max_y=0.5, micpos_range_z=[0.6, 0.9], 
                 corner_mic_dist_max_x=0.03, corner_mic_dist_max_y=0.03, 
                 spkr_mic_dist_range_x=[0.5, 4], spkr_mic_dist_range_y=[0.5, 4], spkr_mic_dist_range_z=[0.1, 0.5], 
                 t60_range = [0.1, 0.4], 
                 min_angle_diff = 30, 
                 max_angle_diff = 360, 
                 micarray='circular7'):

        self._sound_velocity = sound_velocity
        self._fs = fs

        self._roomdim_range_x = roomdim_range_x
        self._roomdim_range_y = roomdim_range_y
        self._roomdim_range_z = roomdim_range_z

        self._micpos = micpos
        self._roomcenter_mic_dist_max_x = roomcenter_mic_dist_max_x
        self._roomcenter_mic_dist_max_y = roomcenter_mic_dist_max_y
        self._micpos_range_z = micpos_range_z

        self._corner_mic_dist_max_x = corner_mic_dist_max_x
        self._corner_mic_dist_max_y = corner_mic_dist_max_y

        self._spkr_mic_dist_range_x = spkr_mic_dist_range_x
        self._spkr_mic_dist_range_y = spkr_mic_dist_range_y
        self._spkr_mic_dist_range_z = spkr_mic_dist_range_z

        self._t60_range = t60_range

        self._min_angle_diff = min_angle_diff
        self._max_angle_diff = max_angle_diff

        if self._min_angle_diff >= self._max_angle_diff:
            raise ValueError('min_angle_diff (given: {}) must be smaller than max_angle_diff (given: {}).'.format(self._min_angle_diff, self._max_angle_diff))

        # microphone array geometry
        if micarray == 'circular7':
            self._micarray = np.concatenate([np.zeros((1,3)), np.array([0.0425 * np.array([np.cos(i * np.pi/3), np.sin(i * np.pi/3), 0]) for i in range(6)])])  # 7x3 array
        elif micarray == 'mono':
            self._micarray = np.zeros((1,3))  # 1x3 array
        elif micarray == 'custom_linear6':
            num_mics = 6
            #mic_spacing = 0.035  # 35mm
            #mic_array_start = np.array([0.035, 0, 0])  # Start position along the wall
            #self._micarray = np.array([mic_array_start + np.array([mic_spacing * i, 0, 0]) for i in range(num_mics)])
            self._micarray = np.array([[i * 0.035, 0, 0] for i in range(6)])
        elif micarray == 'custom4':
            self._micarray = np.array([[0.0575, 0.162, 0],
                                       [0.024, 0, 0],
                                       [0.047, 0, 0],
                                       [0.025, 0.129, 0.0075]])
        self.micarray = micarray

        # print('Instantiating {}'.format(self.__class__.__name__))
        # print('Sound velocity: {}'.format(self._sound_velocity))
        # print('Sampling frequency: {}'.format(self._fs))

        # print('Room dimension range (x): {}'.format(self._roomdim_range_x))
        # print('Room dimension range (y): {}'.format(self._roomdim_range_y))
        # print('Room dimension range (z): {}'.format(self._roomdim_range_z))

        # print('Mic positioning scheme: {}'.format(self._micpos))

        # if self._micpos == 'center':
        #     print('Max distance between room center and mic (x): {}'.format(self._roomcenter_mic_dist_max_x))
        #     print('Max distance between room center and mic (y): {}'.format(self._roomcenter_mic_dist_max_y))
        #     print('Mic position range (z): {}'.format(self._micpos_range_z))
        # elif self._micpos == 'corner':
        #     print('Max distance between room corner and mic (x): {}'.format(self._corner_mic_dist_max_x))
        #     print('Max distance between room corner and mic (y): {}'.format(self._corner_mic_dist_max_y))
        #     print('Mic position range (z): {}'.format(self._micpos_range_z))


        # print('Speaker-mic distance range (x): {}'.format(self._spkr_mic_dist_range_x))
        # print('Speaker-mic distance range (y): {}'.format(self._spkr_mic_dist_range_y))
        # print('Speaker-mic distance range (z): {}'.format(self._spkr_mic_dist_range_z))

        # print('T60 range (z): {}'.format(self._t60_range))

        # print('Minimum angle difference between two sources: {}'.format(self._min_angle_diff))
        # print('Maximum angle difference between two sources: {}'.format(self._max_angle_diff))

        # print('Mic array geometry: {}'.format(micarray))

        # print('', flush=True)



    def __call__(self, nspeakers=2, info_as_display_style=False, device=torch.device('cpu')):
        success = False
        
        while not success:
            # Randomly sample room dimensions. 
            # L = np.array([np.random.uniform(*self._roomdim_range_x), 
            x0, x1 = self._roomdim_range_x
            L = np.array([x0 + (x1 - x0) * np.sqrt(np.random.rand()),
                          np.random.uniform(*self._roomdim_range_y), 
                          np.random.uniform(*self._roomdim_range_z)])
            # L = np.array([5.2, 4.6, 2.85])

            # Randomly sample T60. 
            # rt = np.random.uniform(*self._t60_range)
            # [0.25,0.35] : [0.55,0.65] = 5 : 2
            i = np.random.rand() * 7 - 5
            if i >= 0:
                rt = 0.55 + i / 2 * 0.1
            else:
                rt = 0.35 + i / 5 * 0.1
            # rirlen = int(rt * self._fs)

            # validity check
            V = np.prod(L)
            S = 2 * (L[0]*L[2] + L[1]*L[2] + L[0]*L[1])
            alpha = 24 * V * np.log(10) / (self._sound_velocity * S * rt)
            if alpha < 1:
                success = True

        # Randomly sample a mic array location. 
        if self._micpos == 'center':
            room_center = L / 2
            r = np.array([np.random.uniform(room_center[0] - self._roomcenter_mic_dist_max_x, room_center[0] + self._roomcenter_mic_dist_max_x),
                          np.random.uniform(room_center[1] - self._roomcenter_mic_dist_max_y, room_center[1] + self._roomcenter_mic_dist_max_y),
                          np.random.uniform(*self._micpos_range_z)])
            r = np.maximum(r, 0)
            r = np.minimum(r, L)
            R = self._micarray + r

        elif self._micpos == 'corner':
            corner_x = 'origin' if np.random.choice([0, 1]) == 1 else 'end'
            corner_y = 'origin' if np.random.choice([0, 1]) == 1 else 'end'
            r = np.array([np.random.uniform(0.0425, self._corner_mic_dist_max_x) if corner_x == 'origin'
                          else np.random.uniform(L[0] - self._corner_mic_dist_max_x, L[0] - 0.0425), 
                          np.random.uniform(0.0425, self._corner_mic_dist_max_y) if corner_y == 'origin'
                          else np.random.uniform(L[1] - self._corner_mic_dist_max_y, L[1] - 0.0425), 
                          np.random.uniform(*self._micpos_range_z)])
            r = np.maximum(r, 0)
            r = np.minimum(r, L)
            R = self._micarray + r
        elif self._micpos == 'linear':
            r = np.array([np.random.uniform(L[0]/2-0.8, L[0]/2+0.8), np.random.uniform(L[1]-0.1, L[1]-0.3), np.random.uniform(*self._micpos_range_z)])
            r = np.maximum(r, 0)
            r = np.minimum(r, L)
            tmp = self._micarray
            if self.micarray == 'custom4':
                tmp = [np.array([[0.0575, 0.162, 0], # vertical
                                 [0.024, 0, 0],
                                 [0.047, 0, 0],
                                 [0.025, 0.129, 0.0075]]),
                       np.array([[0, 0.0335, 0], # rotate left
                                 [0.162, 0, 0],
                                 [0.162, 0.023, 0],
                                 [0.033, 0.001, 0.0075]]),
                       np.array([[0.162, 0, 0], # rotate right
                                 [0, 0.0335, 0],
                                 [0, 0.0105, 0],
                                 [0.129, 0.0325, 0.0075]])][np.random.randint(3)]
                theta = np.random.uniform(0, 2 * np.pi)
                rotation_matrix_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                                              [np.sin(theta), np.cos(theta), 0],
                                              [0, 0, 1]])
                tmp = np.dot(tmp, rotation_matrix_z.T)
            R = tmp + r
            #raise ValueError('micpos must be either center or corner: {}'.format(self._micpos))


        # # Randomly sample an ellipse on which sources will be located. 
        # ellipse_xaxis = np.random.uniform(*self._spkr_mic_dist_range_x)
        # ellipse_yaxis = np.random.uniform(*self._spkr_mic_dist_range_y)

        # # Randomly sample a base height. 
        # base_height = np.random.uniform(*self._spkr_mic_dist_range_z)

        mic2src_vecs = []
        speakers = []
        h = []
        for i in range(nspeakers):
            # max_trials = 1000
            
            # for trial in range(max_trials):


            s = np.array([np.random.uniform(0.5, L[0]-0.5), np.random.uniform(0.1, 0.5), np.random.uniform(1, 1.8)])
            s = np.maximum(s, 0)
            s = np.minimum(s, L)
                
            #     valid = True
            #     for spk in speakers:
            #         if np.sqrt(np.sum(np.square(s[:2] -spk[:2]))) < 0.3:
            #             valid = False
            #     if valid:
            #         break

            # if not valid:
            #     raise RuntimeError('Failed to generate valid RIRs.')
                
            speakers.append(s)

            # h0 = np.array(pyrirgen.generateRir(L, s, R, soundVelocity=self._sound_velocity, fs=self._fs, reverbTime=rt, nSamples=rirlen))

       
            # h.append(h0) # [spk, np([mic, rir])]

        speakers = np.array(speakers)
        rir, rir_direct = FRAM_RIR(R, self._fs, rt, room_dim=L, src_pos=speakers, device=device)
                
        # print('Room dimensions: [{:6.3f} m, {:6.3f} m, {:6.3f} m]'.format(L[0], L[1], L[2]))
        # print('T60: {:6.3f} s'.format(rt))
        # print('Mic: {:}'.format(R))
        
        # for i in range(nspeakers):
        #     print('Speaker {}: speaker_location = {:6.3f} m, {:6.3f} m, {:6.3f} m]'.format(i, speakers[i][0], speakers[i][1], speakers[i][2]))
        # print('', flush=True)

        # return
        if info_as_display_style:
            info = [('t60', rt)]
            return rir, rir_direct, info
        else:
            return rir, rir_direct

if __name__ == '__main__':
    
    RandomRirGenerator_kwargs = {
        'fs': 16000,
        't60_range': [ 0.15, 0.7 ],
        'roomdim_range_x': [ 3.2, 5.2 ],
        'roomdim_range_y': [ 2.5, 4.6 ],
        'roomdim_range_z': [ 2.54, 2.85 ],
        'micpos_range_z': [ 0.8, 0.8 ],
        'micpos': 'linear',
        'micarray': 'custom_linear6'
    }
    
    rirgen = RandomRirGenerator(**RandomRirGenerator_kwargs)
    start = time()
    rir, rir_direct = rirgen() # [M, spk, L]
    end = time()
    # print('rir', rir.shape, 'rir direct', rir_direct.shape)
    near,_ = sf.read('near.wav')
    far = []
    anec = []
    for i in range(6):
        far.append(scipy.signal.lfilter(rir[i][0], 1, near))
        anec.append(scipy.signal.lfilter(rir_direct[i][0], 1, near))
    print('rir time', end - start)
    print('conv time', time() - end)
    far = np.array(far) / np.max(far) * np.max(near)
    anec = np.array(anec) / np.max(anec) * np.max(near)
    # print('far', far.shape, 'anec', anec.shape)
    sf.write('far.wav', far.transpose((1,0)), 16000)
    sf.write('anechoic.wav', anec.transpose((1,0)), 16000)