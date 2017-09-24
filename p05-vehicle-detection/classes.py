import numpy as np


class FeatureExtractionParameters:
    def __init__(self, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
                 cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True, shape=(64, 64),
                 cells_per_step=2):
        self.color_space = color_space  # RGB, HSV, LUV, HLS, YUV, YCrCb
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel  # Can be 0, 1, 2, or "ALL"
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.shape = shape
        self.cells_per_step = cells_per_step


class BoxesDetectedWithCars:
    def __init__(self):
        self.index = 0
        self.max_frames_accumulated = 25
        self.boxes_frame = np.empty(self.max_frames_accumulated, dtype=object)

    def add_boxes(self, boxes):
        self.boxes_frame[self.index] = boxes
        self.index += 1
        self.index %= self.max_frames_accumulated

    def get_boxes(self):
        cars = []
        for boxes in self.boxes_frame:
            if boxes is not None:
                for box in boxes:
                    cars.append(box)

        return cars
