from collections import deque, Counter
from src.config import SMOOTHING_WINDOW


class TemporalSmoother:
    def __init__(self, window_size=SMOOTHING_WINDOW, confirmation_frames=3):
        """
        Stabilizes predictions using temporal confirmation.

        A prediction becomes stable when it is either:
        1. observed consecutively for confirmation_frames, or
        2. the majority prediction in the recent history window.
        """
        self.window_size = window_size
        self.confirmation_frames = confirmation_frames
        self.history = deque(maxlen=window_size)

        self.candidate = None
        self.candidate_count = 0
        self.stable_prediction = None

    def update(self, prediction):
        """Update temporal state and return the stable prediction."""
        if prediction is None:
            return self.stable_prediction

        self.history.append(prediction)

        if prediction == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = prediction
            self.candidate_count = 1

        # Fast path: consistent predictions are accepted quickly.
        if self.candidate_count >= self.confirmation_frames:
            self.stable_prediction = self.candidate

        # Fallback: preserve majority-based stabilization.
        counter = Counter(self.history)
        most_common_prediction, count = counter.most_common(1)[0]

        if count >= (len(self.history) + 1) // 2:
            self.stable_prediction = most_common_prediction

        return self.stable_prediction

    def get_stable_prediction(self):
        """Returns the currently confirmed prediction."""
        return self.stable_prediction

    def clear(self):
        """Clears all temporal state when the hand disappears."""
        self.history.clear()
        self.candidate = None
        self.candidate_count = 0
        self.stable_prediction = None
