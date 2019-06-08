# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from . import run_mhkf
import pdb

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

        self.stashed_buffer ={} #( track_id:track)
        self.stashed_tracks=[] #tracks
        self.stashed_track_id=[]
        self.prev_archive_size=-1
        self.stashed_features={}

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections, revived = \
            self._match(detections)


            # (List[(int, int)], List[int], List[int])
            #     Returns a tuple with the following three entries:
            #     * A list of matched track and detection indices.
            #     * A list of unmatched track indices.
            #     * A list of unmatched detection indices.
        # pdb.set_trace()
        if len(revived)>0:
            for track_id, detection_idx in revived:
                # pdb.set_trace()
                self.revive(self.stashed_buffer[track_id],detections[detection_idx])

                # self.stashed_buffer.pop(track_id)
        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            # if track.track_id == 131:
            #     pdb.set_trace()
            features += track.features
            targets += [track.track_id for _ in track.features]
            if track.track_id != 131:
                track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            return cost_matrix

        def zombie_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            # pdb.set_trace()
            stashed_features = np.array([tracks[i].features for i in track_indices])
            # = np.array([tracks[i].track_id for i in track_indices])
            # print('hey')
            cost_matrix = self.metric.distance_against_saved_features(features, stashed_features)
            #skips Kalman filtering for zombie tracks.

            # cost_matrix = linear_assignment.gate_cost_matrix(
            #     self.kf, cost_matrix, tracks, dets, track_indices,
            #     detection_indices)
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        if len(self.stashed_buffer.keys()) != self.prev_archive_size: # if the size has changed
            # print('------------------')
            # print('%d + %d =' % (len(self.tracks), len(self.stashed_buffer)))
            # pdb.set_trace()
            # self.stashed_tracks = [track_tuple[0] for track_tuple in self.stashed_buffer]
            # self.stashed_track_id = [track_tuple[1] for track_tuple in self.stashed_buffer]
            self.stashed_track_id =self.stashed_buffer.keys()
            self.stashed_track =self.stashed_buffer.values()
            # self.tracks =list(set(self.tracks+stashed_tracks))
            # print(len(self.tracks))
            # print('------------------')
            self.prev_archive_size=len(self.stashed_buffer.keys())

        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]


# reidentified_tracks, still_unmatched_tracks, unmatched_detections =linear_assignment.matching_cascade(    zombie_metric, float('inf'), self.max_age,    self.stashed_tracks, detections)
        # pdb.set_trace()
        # unconfirmed_tracks = unconfirmed_tracks + self.stashed_buffer
        # pdb.set_trace()
        reidentified_tracks = []
        if len(self.stashed_buffer.keys())>0:
        #force matching for the stashed tracks.
            # print('trying....')
            reidentified_tracks, still_unmatched_tracks, unmatched_detections = \
                linear_assignment.reviving_cascade(
                #if we're trying to revive the old track,
                #run zombiemetric instead of normal metric.
                    zombie_metric, float('inf'), 30,
                    self.stashed_buffer, detections) #getting some errors here but other than that i'm ok
            # for j in range(len(reidentified_tracks)):
            #     tracky = reidentified_tracks[j]
            #     tracky_idx = tracky[1]
            # pdb.set_trace()
                # if len(detections)>1:
                #     detections = detections[:tracky_idx] + detections[tracky_idx+1:]
                # else:
                #     detections = []

        #assuming that the key detection won't be picked by the other tracks for now.
        #either just need to send out the indices or return the detections

        # pdb.set_trace()
        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
                            k for k in unmatched_tracks_a if
                            self.tracks[k].time_since_update == 1]
        # lost tracks
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        #unmatched even after iou matching the detections
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        # attempt for re-identification;
        #need to save only the first time:
        # inefficient for sure but will work for JR setting
        # pdb.set_trace()
        if unmatched_tracks_a != [] :
            if len(self.stashed_buffer.keys()) <=20 :
                for idx in unmatched_tracks_a:

                    tracky = self.tracks[idx]
                    if tracky.track_id>=120: #debug
                        # if tracky.track_id == 131:
                        #     pdb.set_trace() #around 284
                        if tracky.track_id not in self.stashed_buffer and tracky.features:

                            self.stashed_buffer[tracky.track_id]= tracky
                            print('saving tack \# %d'%tracky.track_id)

        matches = matches_a + matches_b #+ reidentified_tracks
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections, reidentified_tracks

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1

    def revive(self, track, detection):
        """
        revive the dead track back to life.
        """
        print('revivded track %d'%track.track_id)
        # pdb.set_trace()
        mean, covariance = self.kf.initiate(detection.to_xyah())
        # self.tracks.append(Track(
        #     mean, covariance, self._next_id, self.n_init, self.max_age,
        #     detection.feature))
        #next id?
        track.time_since_update = 0
        track.age = 1
        track.hits = 1
        track.features.append(detection.feature)
        track.state = 2
        self.tracks.append(track)
        self.stashed_buffer.pop(track.track_id)
        # pdb.set_trace()
