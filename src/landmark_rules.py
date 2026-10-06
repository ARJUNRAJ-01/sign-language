import numpy as np
import math

class LandmarkRuleEngine:
    """
    Rotation-invariant geometric rule engine using MediaPipe 21 Hand Landmarks
    to eliminate visual ambiguities between similar ASL alphabet gestures.
    """
    
    @staticmethod
    def get_finger_states(landmarks_coords):
        """
        Determines the extension and curl state of each finger using rotation-invariant distances.
        landmarks_coords: list of 21 (x, y) tuples
        """
        if len(landmarks_coords) < 21:
            return None
            
        points = [np.array(p, dtype=float) for p in landmarks_coords]
        wrist = points[0]
        
        # Finger base, pip, tip
        thumb_cmc, thumb_mcp, thumb_ip, thumb_tip = points[1], points[2], points[3], points[4]
        index_mcp, index_pip, index_dip, index_tip = points[5], points[6], points[7], points[8]
        mid_mcp, mid_pip, mid_dip, mid_tip = points[9], points[10], points[11], points[12]
        ring_mcp, ring_pip, ring_dip, ring_tip = points[13], points[14], points[15], points[16]
        pinky_mcp, pinky_pip, pinky_dip, pinky_tip = points[17], points[18], points[19], points[20]
        
        def dist(p1, p2):
            return float(np.linalg.norm(p1 - p2))
            
        def get_angle(p1, p2, p3):
            # Angle at p2 (p1-p2-p3)
            v1 = p1 - p2
            v2 = p3 - p2
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 == 0 or norm_v2 == 0: return 0.0
            cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
            
        # Extension based on tip being further from wrist than PIP (robust to rotation)
        # Using 1.05 to be slightly more forgiving for extended fingers
        index_ext = dist(index_tip, wrist) > dist(index_pip, wrist) * 1.05
        mid_ext = dist(mid_tip, wrist) > dist(mid_pip, wrist) * 1.05
        ring_ext = dist(ring_tip, wrist) > dist(ring_pip, wrist) * 1.05
        pinky_ext = dist(pinky_tip, wrist) > dist(pinky_pip, wrist) * 1.05
        
        # Thumb is extended if its tip is further from the pinky base than its IP joint
        # This accurately captures "Y" (thumb out) vs "A" (thumb tucked in against index)
        thumb_ext = dist(thumb_tip, pinky_mcp) > dist(thumb_ip, pinky_mcp) * 1.1
        
        # Is the thumb folded across the palm?
        thumb_across_palm = dist(thumb_tip, pinky_mcp) < dist(index_mcp, pinky_mcp)
        
        # Are index and middle fingers spread apart? (For V vs U/K)
        fingers_spread = dist(index_tip, mid_tip) > dist(index_mcp, mid_mcp) * 1.5
        
        # Horizontal direction check for index pointing (for G vs D)
        index_vec = index_tip - index_mcp
        is_horizontal = abs(index_vec[0]) > abs(index_vec[1]) * 1.2
        
        return {
            "thumb": thumb_ext,
            "index": index_ext,
            "middle": mid_ext,
            "ring": ring_ext,
            "pinky": pinky_ext,
            "horizontal": is_horizontal,
            "thumb_across": thumb_across_palm,
            "spread": fingers_spread,
            "thumb_tip": thumb_tip,
            "index_tip": index_tip,
            "mid_tip": mid_tip,
            "ring_tip": ring_tip,
            "pinky_tip": pinky_tip,
            "index_mcp": index_mcp,
            "mid_mcp": mid_mcp,
            "wrist": wrist,
            "dist": dist,
            "angle": get_angle
        }

    @classmethod
    def disambiguate(cls, model_pred, model_conf, landmarks_coords, top_candidates):
        """
        Refines model predictions based on anatomical hand geometric rules.
        """
        if not landmarks_coords or len(landmarks_coords) < 21:
            return model_pred, model_conf

        states = cls.get_finger_states(landmarks_coords)
        if states is None:
            return model_pred, model_conf

        thumb = states["thumb"]
        idx = states["index"]
        mid = states["middle"]
        ring = states["ring"]
        pinky = states["pinky"]
        horiz = states["horizontal"]
        thumb_across = states["thumb_across"]
        spread = states["spread"]
        d = states["dist"]
        
        tt = states["thumb_tip"]
        it = states["index_tip"]
        mt = states["mid_tip"]
        pt = states["pinky_tip"]
        wrist = states["wrist"]

        # 1. Disambiguate 'Y' (Hang loose / Shaka sign)
        # Thumb extended AND Pinky extended, middle 3 fingers (index, mid, ring) curled
        if pinky and thumb and not idx and not mid and not ring:
            return "Y", max(model_conf, 0.92)
            
        # 2. Disambiguate 'I' (Pinky only, thumb folded)
        if pinky and not thumb and not idx and not mid and not ring:
            return "I", max(model_conf, 0.90)

        # 3. Disambiguate 'A', 'S', 'E', 'M', 'N', 'T' (Fists)
        if not pinky and not idx and not mid and not ring:
            if thumb and not thumb_across:
                # Thumb is out on the side but not pinky -> "A"
                if model_pred in ["A", "Y", "I", "S", "E", "T", "M", "N"]:
                    return "A", max(model_conf, 0.88)
            elif thumb_across:
                # Thumb folded across the fingers
                if model_pred in ["S", "E", "M", "N", "T", "A"]:
                    if tt[1] > it[1] + 15: # Thumb is below the curled index (y goes down in images)
                        return "E", max(model_conf, 0.86)
                    elif d(tt, it) < 32: # Thumb near index -> "T"
                        return "T", max(model_conf, 0.85)
                    elif d(tt, mt) < 32: # Thumb near middle -> "N"
                        return "N", max(model_conf, 0.85)
                    elif d(tt, states["ring_tip"]) < 32: # Thumb near ring -> "M"
                        return "M", max(model_conf, 0.85)
                    else:
                        return "S", max(model_conf, 0.88) # Default fist

        # 4. Disambiguate 'D' vs 'G' vs 'Z'
        # 'D': Index straight UP (vertical)
        # 'G': Index pointed HORIZONTALLY
        # 'Z': Like 'D' but motion-based. If model strongly predicts Z, allow it.
        if idx and not mid and not ring and not pinky and model_pred in ["D", "G", "Z"]:
            if horiz or model_pred == "G":
                return "G", max(model_conf, 0.88)
            elif model_pred == "Z":
                return "Z", max(model_conf, 0.90)
            else:
                return "D", max(model_conf, 0.90)

        # 5. Disambiguate 'F' vs 'W'
        if mid and ring and model_pred in ["F", "W"]:
            if pinky and d(it, tt) < 55: # index touches thumb (circle), other 3 extended
                return "F", max(model_conf, 0.92)
            if idx and not pinky: # index, middle, ring extended. pinky down
                return "W", max(model_conf, 0.92)

        # 6. Disambiguate 'H' vs 'B' vs 'U' vs 'V' vs 'K'
        if idx and mid and model_pred in ["H", "B", "U", "V", "K"]:
            if ring and pinky:
                # All 4 fingers up -> B
                return "B", max(model_conf, 0.92)
            if not ring and not pinky:
                if horiz:
                    # 2 fingers horizontal -> H
                    return "H", max(model_conf, 0.90)
                else:
                    if spread:
                        # V or K
                        # In 'K', the thumb is between the index and middle fingers, pointing up
                        if thumb and d(tt, states["index_mcp"]) < 50:
                            return "K", max(model_conf, 0.88)
                        else:
                            return "V", max(model_conf, 0.90)
                    else:
                        return "U", max(model_conf, 0.90)

        return model_pred, model_conf
