from eolearn.io import S2L1CWCSInput, S2L2AWCSInput
from eolearn.core import FeatureType, EOTask
from sentinelhub import BBox, CRS
from sentinelhub.download import DownloadFailedException
from requests.exceptions import HTTPError
import numpy as np
import datetime
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class SimpleFilterTask(EOTask):
    """
    Transforms an eopatch of shape [n, w, h, d] into [m, w, h, d] for m <= n. It removes all slices which don't
    conform to the filter_func.
    A filter_func is a callable which takes an numpy array and returns a bool.
    """
    def __init__(self, feature, filter_func, filter_features=...):
        """
        :param feature: Feature in the EOPatch , e.g. feature=(FeatureType.DATA, 'bands')
        :type feature: (FeatureType, str)
        :param filter_func: A callable that takes a numpy evaluates to bool.
        :type filter_func: object
        :param filter_features: A collection of features which will be filtered
        :type filter_features: dict(FeatureType: set(str))
        """
        self.feature = self._parse_features(feature)
        self.filter_func = filter_func
        self.filter_features = self._parse_features(filter_features)

    def _get_filtered_indices(self, feature_data):
        return [idx for idx, img in enumerate(feature_data) if self.filter_func(img)]

    def _update_other_data(self, eopatch):
        pass

    def execute(self, eopatch):
        """
        :param eopatch: Input EOPatch.
        :type eopatch: EOPatch
        :return: Transformed eo patch
        :rtype: EOPatch
        """
        feature_type, feature_name = next(self.feature(eopatch))

        good_idxs = self._get_filtered_indices(eopatch[feature_type][feature_name] if feature_name is not ... else
                                               eopatch[feature_type])

        for feature_type, feature_name in self.filter_features(eopatch):
            if feature_type.is_time_dependent():
                if feature_type.has_dict():
                    if feature_type.contains_ndarrays():
                        eopatch[feature_type][feature_name] = np.asarray([eopatch[feature_type][feature_name][idx] for
                                                                          idx in good_idxs])
                    # else:
                    #     NotImplemented
                else:
                    eopatch[feature_type] = [eopatch[feature_type][idx] for idx in good_idxs]

        eopatch.timestamp = [eopatch.timestamp[idx] for idx in good_idxs]

        self._update_other_data(eopatch)

        return eopatch


class ValidDataFractionPredicate:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        return np.sum(array)/np.prod(array.shape) >= self.threshold


class SelectRandomFrameTask(EOTask):
    def __init__(self, features, n_frames=1):
        self.features = self._parse_features(features)
        self.n_frames = n_frames

    def execute(self, eopatch):
        to_keep = np.random.choice(len(eopatch.timestamp), self.n_frames)
        eopatch.timestamp = [eopatch.timestamp[tk] for tk in to_keep]
        for feature_type, feature_name in self.features(eopatch):
            eopatch[feature_type][feature_name] = eopatch[feature_type][feature_name][to_keep]
        return eopatch


class S2L1CToL2AWorkflow:
    def __init__(self, config):
        self.config = config

    def _get_random_bbox(self):
        lat = self.config.lat_bot_right + (self.config.lat_top_left - self.config.lat_bot_right) * np.random.ranf(1)
        lon = self.config.lon_top_left + (self.config.lon_bot_right - self.config.lon_top_left) * np.random.ranf(1)
        return BBox(((lon, lat), (lon + self.config.delta_lon, lat - self.config.delta_lat)), crs=CRS.WGS84)

    def _get_eopatch(self, bbox, time_interval):
        # request L2A images
        add_l2a = S2L2AWCSInput(self.config.l2a_field,
                                resx=self.config.resx,
                                resy=self.config.resy,
                                maxcc=self.config.maxcc,
                                time_difference=datetime.timedelta(hours=2))
        # filter frames with valid data = 1
        filter_task = SimpleFilterTask((FeatureType.MASK, 'IS_DATA'), ValidDataFractionPredicate(1),
                                       (FeatureType.DATA, self.config.l2a_field))
        # randomly select one temporal frame
        sel_task = SelectRandomFrameTask([(FeatureType.DATA, self.config.l2a_field),
                                          (FeatureType.MASK, 'IS_DATA')])
        # request corresponding L1C
        add_l1c = S2L1CWCSInput(self.config.l1c_field,
                                resx=self.config.resx,
                                resy=self.config.resy,
                                maxcc=self.config.maxcc,
                                time_difference=datetime.timedelta(hours=2))
        # execute workflow
        try:
            eop = add_l1c.execute(
                sel_task.execute(
                    filter_task.execute(
                        add_l2a.execute(time_interval=time_interval, bbox=bbox))))
            executed = True
        except (HTTPError, IndexError, DownloadFailedException, RuntimeError, ValueError, TypeError, NameError):
            logging.error("Exception thrown in obtaining EOPatch")
            eop = None
            executed = False
        return eop, executed

    def execute(self):
        eop, executed = self._get_eopatch(self._get_random_bbox(), (self.config.start_time, self.config.end_time))
        return eop, executed
