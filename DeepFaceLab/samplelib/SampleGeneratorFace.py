import multiprocessing
import time
import traceback

import cv2
import numpy as np

from core import mplib
from core.interact import interact as io
from core.joblib import SubprocessGenerator, ThisThreadGenerator
from facelib import LandmarksProcessor
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor,
                       SampleType)


'''
arg
output_sample_types = [
                        [SampleProcessor.TypeFlags, size, (optional) {} opts ] ,
                        ...
                      ]
'''
class SampleGeneratorFace(SampleGeneratorBase):
    def __init__ (self, samples_path, debug=False, batch_size=1,
                        random_ct_samples_path=None,
                        sample_process_options=SampleProcessor.Options(),
                        output_sample_types=[],
                        uniform_yaw_distribution=False,
                        generators_count=4,
                        raise_on_no_data=True,                        
                        **kwargs):

        super().__init__(debug, batch_size)
        self.initialized = False
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types
        
        if self.debug:
            self.generators_count = 1
        else:
            self.generators_count = max(1, generators_count)

        samples = SampleLoader.load (SampleType.FACE, samples_path)
        self.samples_len = len(samples)
        
        if self.samples_len == 0:
            if raise_on_no_data:
                raise ValueError('No training data provided.')
            else:
                return
                
        if uniform_yaw_distribution:  # Now it means uniform distribution for both pitch and yaw 都优化
            samples_pyr = [(idx, sample.get_pitch_yaw_roll()) for idx, sample in enumerate(samples)]
            
            grads = 128
            pitch_space = np.linspace(-1.2, 1.2, grads)  # Assuming the same range for pitch
            yaw_space = np.linspace(-1.2, 1.2, grads)  # Yaw range

            grid_samples_list = [[None]*grads for _ in range(grads)]
            for g_pitch in io.progress_bar_generator(range(grads), "侧脸和上下都优化"):
                pitch = pitch_space[g_pitch]
                next_pitch = pitch_space[g_pitch+1] if g_pitch < grads-1 else pitch + (pitch_space[1] - pitch_space[0])
                for g_yaw in range(grads):
                    yaw = yaw_space[g_yaw]
                    next_yaw = yaw_space[g_yaw+1] if g_yaw < grads-1 else yaw + (yaw_space[1] - yaw_space[0])
                    
                    grid_samples = []
                    for idx, pyr in samples_pyr:
                        s_pitch, s_yaw, _ = pyr
                        s_pitch, s_yaw = -s_pitch, -s_yaw  # Adjusting the signs if necessary
                        
                        if (pitch <= s_pitch < next_pitch) and (yaw <= s_yaw < next_yaw):
                            grid_samples.append(idx)
                    if len(grid_samples) > 0:
                        if grid_samples_list[g_pitch][g_yaw] is None:
                            grid_samples_list[g_pitch][g_yaw] = []
                        grid_samples_list[g_pitch][g_yaw].extend(grid_samples)
            
            # Flatten the list and filter out None values
            yaws_sample_list = [item for sublist in grid_samples_list for item in sublist if item is not None]

            index_host = mplib.Index2DHost(yaws_sample_list)
        else:
            index_host = mplib.IndexHost(self.samples_len)

        if random_ct_samples_path is not None:
            ct_samples = SampleLoader.load (SampleType.FACE, random_ct_samples_path)
            ct_index_host = mplib.IndexHost( len(ct_samples) )
        else:
            ct_samples = None
            ct_index_host = None

        if self.debug:
            self.generators = [ThisThreadGenerator ( self.batch_func, (samples, index_host.create_cli(), ct_samples, ct_index_host.create_cli() if ct_index_host is not None else None) )]
        else:
            self.generators = [SubprocessGenerator ( self.batch_func, (samples, index_host.create_cli(), ct_samples, ct_index_host.create_cli() if ct_index_host is not None else None), start_now=False ) \
                               for i in range(self.generators_count) ]
                               
            SubprocessGenerator.start_in_parallel( self.generators )

        self.generator_counter = -1
        
        self.initialized = True
        
    #overridable
    def is_initialized(self):
        return self.initialized
        
    def __iter__(self):
        return self

    def __next__(self):
        if not self.initialized:
            return []
            
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, param ):
        samples, index_host, ct_samples, ct_index_host = param
 
        bs = self.batch_size
        while True:
            batches = None

            indexes = index_host.multi_get(bs)
            ct_indexes = ct_index_host.multi_get(bs) if ct_samples is not None else None

            t = time.time()
            for n_batch in range(bs):
                sample_idx = indexes[n_batch]
                sample = samples[sample_idx]

                ct_sample = None
                if ct_samples is not None:
                    ct_sample = ct_samples[ct_indexes[n_batch]]

                try:
                    x, = SampleProcessor.process ([sample], self.sample_process_options, self.output_sample_types, self.debug, ct_sample=ct_sample)
                except:
                    raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )

                if batches is None:
                    batches = [ [] for _ in range(len(x)) ]

                for i in range(len(x)):
                    batches[i].append ( x[i] )

            yield [ np.array(batch) for batch in batches]
